//! vec_index.rs — índice de vectores en disco, vía `usearch` (HNSW),
//! PARTICIONADO en varios shards POR BUNDLE.
//!
//! Cada bundle tiene su propia carpeta bajo el directorio base
//! (ej. vectors/wikipedia/, vectors/plugins/) y dentro de cada una
//! sus propios shards (vectors.0.tvim .. vectors.15.tvim).
//! Esto permite borrar un bundle entero simplemente eliminando su
//! carpeta, y aisla físicamente los índices sin perder el sharding
//! que mitiga usearch#514.

use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub const DIMENSIONS: usize = crate::embed::EMBED_DIM;
const NUM_SHARDS: usize = 16;

fn options() -> IndexOptions {
    IndexOptions {
        dimensions: DIMENSIONS,
        metric: MetricKind::Cos,
        quantization: ScalarKind::I8,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        multi: false,
    }
}

fn shard_path(dir: &Path, i: usize) -> PathBuf {
    dir.join(format!("vectors.{i}.tvim"))
}

struct Shard {
    index: Index,
    path: PathBuf,
    loaded: bool,
    dirty: bool,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, String> {
        let index =
            Index::new(&options()).map_err(|e| format!("Error creando shard usearch: {e}"))?;
        let loaded = if path.exists() {
            let path_str = path.to_str().ok_or("Ruta de shard inválida")?;
            index
                .view(path_str)
                .map_err(|e| format!("Error abriendo (view) shard {}: {e}", path.display()))?;
            false
        } else {
            true
        };
        Ok(Shard {
            index,
            path: path.to_path_buf(),
            loaded,
            dirty: false,
        })
    }

    fn ensure_loaded(&mut self) -> Result<(), String> {
        if self.loaded {
            return Ok(());
        }
        let fresh =
            Index::new(&options()).map_err(|e| format!("Error recreando shard usearch: {e}"))?;
        if self.path.exists() {
            let path_str = self.path.to_str().ok_or("Ruta de shard inválida")?;
            fresh
                .load(path_str)
                .map_err(|e| format!("Error cargando shard {}: {e}", self.path.display()))?;
        }
        self.index = fresh;
        self.loaded = true;
        Ok(())
    }

    fn release_ram(&mut self) -> Result<(), String> {
        self.save()?;
        if !self.loaded || !self.path.exists() {
            return Ok(());
        }
        let fresh =
            Index::new(&options()).map_err(|e| format!("Error recreando shard usearch: {e}"))?;
        let path_str = self.path.to_str().ok_or("Ruta de shard inválida")?;
        fresh
            .view(path_str)
            .map_err(|e| format!("Error volviendo a view shard {}: {e}", self.path.display()))?;
        self.index = fresh;
        self.loaded = false;
        Ok(())
    }

    fn reserve(&mut self, capacity: usize) -> Result<(), String> {
        self.ensure_loaded()?;
        self.index
            .reserve(capacity)
            .map_err(|e| format!("Error en reserve de shard: {e}"))
    }

    fn add_batch(&mut self, ids: &[i64], vectors_flat: &[f32]) -> Result<(), String> {
        if ids.is_empty() {
            return Ok(());
        }
        self.ensure_loaded()?;
        let needed = self.index.size() + ids.len();
        if needed > self.index.capacity() {
            let target = ((needed as f64) * 1.5).max((needed + 2_000) as f64) as usize;
            self.index
                .reserve(target)
                .map_err(|e| format!("Error en reserve de shard ({target}): {e}"))?;
        }
        for (i, &id) in ids.iter().enumerate() {
            let start = i * DIMENSIONS;
            let end = start + DIMENSIONS;
            self.index
                .add(id as u64, &vectors_flat[start..end])
                .map_err(|e| format!("Error en add: {e}"))?;
        }
        self.dirty = true;
        Ok(())
    }

    fn remove(&mut self, id: i64) {
        if self.ensure_loaded().is_err() {
            return;
        }
        if self.index.remove(id as u64).is_ok() {
            self.dirty = true;
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        match self.index.search(query, k) {
            Ok(r) => r
                .keys
                .into_iter()
                .zip(r.distances.into_iter())
                .map(|(id, d)| (id as i64, d))
                .collect(),
            Err(_) => vec![],
        }
    }

    fn save(&mut self) -> Result<(), String> {
        if !self.dirty || !self.loaded {
            return Ok(());
        }
        let path_str = self.path.to_str().ok_or("Ruta de shard inválida")?;
        self.index
            .save(path_str)
            .map_err(|e| format!("Error guardando shard {}: {e}", self.path.display()))?;
        self.dirty = false;
        Ok(())
    }
}

struct BundleIndex {
    shards: Vec<Shard>,
}

impl BundleIndex {
    fn open(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Error creando dir de bundle index: {e}"))?;
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for i in 0..NUM_SHARDS {
            shards.push(Shard::open(&shard_path(dir, i))?);
        }
        Ok(BundleIndex { shards })
    }

    fn release_ram(&mut self) -> Result<(), String> {
        for s in &mut self.shards {
            s.release_ram()?;
        }
        Ok(())
    }

    fn reserve(&mut self, capacity: usize) -> Result<(), String> {
        let per_shard = (capacity / NUM_SHARDS).max(1);
        for s in &mut self.shards {
            s.reserve(per_shard)?;
        }
        Ok(())
    }

    fn add_batch(&mut self, ids: &[i64], vectors_flat: &[f32]) -> Result<(), String> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut buckets: Vec<(Vec<i64>, Vec<f32>)> =
            (0..NUM_SHARDS).map(|_| (Vec::new(), Vec::new())).collect();
        for (i, &id) in ids.iter().enumerate() {
            let shard_idx = id.rem_euclid(NUM_SHARDS as i64) as usize;
            buckets[shard_idx].0.push(id);
            buckets[shard_idx]
                .1
                .extend_from_slice(&vectors_flat[i * DIMENSIONS..(i + 1) * DIMENSIONS]);
        }
        for (idx, (shard_ids, shard_vecs)) in buckets.into_iter().enumerate() {
            if !shard_ids.is_empty() {
                self.shards[idx].add_batch(&shard_ids, &shard_vecs)?;
            }
        }
        Ok(())
    }

    fn remove(&mut self, id: i64) {
        let shard_idx = id.rem_euclid(NUM_SHARDS as i64) as usize;
        self.shards[shard_idx].remove(id);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        let mut all: Vec<(i64, f32)> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| scope.spawn(|| shard.search(query, k)))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        });
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }

    fn save(&mut self) -> Result<(), String> {
        for s in &mut self.shards {
            s.save()?;
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.index.size()).sum()
    }
}

pub struct VecIndex {
    base_dir: PathBuf,
    bundles: FxHashMap<String, BundleIndex>,
}

impl VecIndex {
    pub fn open(base_dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(base_dir).map_err(|e| e.to_string())?;
        let mut bundles = FxHashMap::default();
        if let Ok(entries) = std::fs::read_dir(base_dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_dir() {
                    let name = path
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if !name.is_empty() {
                        match BundleIndex::open(&path) {
                            Ok(bi) => {
                                bundles.insert(name, bi);
                            }
                            Err(e) => {
                                crate::plugin_log(&format!(
                                    "WARN no se pudo abrir índice de bundle '{}': {e}",
                                    path.display()
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(VecIndex {
            base_dir: base_dir.to_path_buf(),
            bundles,
        })
    }

    fn bundle_dir(&self, bundle: &str) -> PathBuf {
        self.base_dir.join(bundle)
    }

    fn get_or_create(&mut self, bundle: &str) -> Result<&mut BundleIndex, String> {
        if !crate::frontmatter::valid_bundle_name(bundle) {
            return Err("Nombre de bundle inválido".to_string());
        }
        if !self.bundles.contains_key(bundle) {
            let dir = self.bundle_dir(bundle);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            self.bundles
                .insert(bundle.to_string(), BundleIndex::open(&dir)?);
        }
        Ok(self.bundles.get_mut(bundle).unwrap())
    }

    pub fn add_batch(
        &mut self,
        bundle: &str,
        ids: &[i64],
        vectors_flat: &[f32],
    ) -> Result<(), String> {
        self.get_or_create(bundle)?.add_batch(ids, vectors_flat)
    }

    pub fn remove(&mut self, bundle: &str, id: i64) {
        if let Some(b) = self.bundles.get_mut(bundle) {
            b.remove(id);
        }
    }

    pub fn search(&self, bundle: Option<&str>, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        match bundle {
            Some(b) => self
                .bundles
                .get(b)
                .map(|bi| bi.search(query, k))
                .unwrap_or_default(),
            None => {
                let mut all: Vec<(i64, f32)> = std::thread::scope(|scope| {
                    let handles: Vec<_> = self
                        .bundles
                        .values()
                        .map(|bi| scope.spawn(|| bi.search(query, k)))
                        .collect();
                    handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap_or_default())
                        .collect()
                });
                all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                all.truncate(k);
                all
            }
        }
    }

    pub fn len(&self) -> usize {
        self.bundles.values().map(|b| b.len()).sum()
    }

    pub fn save(&mut self) -> Result<(), String> {
        for b in self.bundles.values_mut() {
            b.save()?;
        }
        Ok(())
    }

    pub fn release_ram(&mut self) -> Result<(), String> {
        for b in self.bundles.values_mut() {
            b.release_ram()?;
        }
        Ok(())
    }

    pub fn reserve(&mut self, bundle: &str, capacity: usize) -> Result<(), String> {
        self.get_or_create(bundle)?.reserve(capacity)
    }

    pub fn remove_bundle_dir(&mut self, bundle: &str) -> Result<(), String> {
        if !crate::frontmatter::valid_bundle_name(bundle) {
            return Err("Nombre de bundle inválido".to_string());
        }
        let dir = self.bundle_dir(bundle);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
        }
        self.bundles.remove(bundle);
        Ok(())
    }

    pub fn wipe(&mut self) -> Result<(), String> {
        self.bundles.clear();
        if self.base_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.base_dir);
        }
        std::fs::create_dir_all(&self.base_dir).map_err(|e| e.to_string())?;
        Ok(())
    }
}
