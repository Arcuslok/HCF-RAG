# HCF-RAG — asistente de escritorio con RAG 100% local

**Proyecto de título · ISI801 Gestión de Proyectos Informáticos · AIEP, Universidad Andrés Bello**

HCF-RAG  es un asistente que responde preguntas usando los documentos del propio usuario, sin que ningún
documento salga de su computador. Se implementa como `ame_rag_plugin`, una DLL nativa escrita en Rust que se
integra al asistente de escritorio A.M.E. (Tauri + Rust).

| | |
|---|---|
| **Equipo** | Alejandro Gutiérrez (director de proyecto, arquitectura, medición y privacidad) · Diego Pérez (ingesta y recuperación) · Matilde Fuentes (interfaz y pruebas) |
| **Docente** | Pablo Hernández Miranda |
| **Documentación de gestión** | `InformeFinal_ISI801_HCF-RAG.pdf` (presupuesto, riesgos, cronograma, KPI y control) |

---

## 1. El problema

Los profesionales independientes acumulan contratos, informes y apuntes, y pierden tiempo buscándolos. Los
asistentes de documentos en la nube resuelven la búsqueda, pero obligan a subir información de terceros
(clientes, pacientes, contrapartes). HCF-RAG propone lo contrario: **indexar y consultar todo en el equipo de la
persona**, incluso en notebooks de 8 GB de RAM sin GPU dedicada.

## 2. Qué hace

Funciones esenciales de la versión 1:

1. **Cargar** documentos de texto y Markdown a una colección (*bundle*).
2. **Buscar** con recuperación híbrida: léxica (BM25 con FTS5) y vectorial, fusionadas en un solo puntaje.
3. **Entregar al modelo solo el contexto relevante**: el hook `on_before_request` busca antes de cada
   pregunta e inyecta los fragmentos con puntaje mínimo de 0,20, condensados para ocupar poco contexto.

Además: grafo de relaciones entre documentos (enlaces explícitos, etiquetas compartidas y similitud),
mapa de similitud, reindexación y reconstrucción en segundo plano.

## 3. Arquitectura

```
Documento (.md / .txt)
      │  ingesta: frontmatter + fragmentación (900 caracteres, 150 de solape)
      ▼
┌───────────────────────────┐      ┌─────────────────────────────┐
│ SQLite (WAL)              │      │ usearch (HNSW, i8, 16 shards)│
│ documents · chunks        │      │ 512 dimensiones, coseno      │
│ chunks_fts (FTS5 + triggers)│    │ clave = chunks.id            │
└─────────────┬─────────────┘      └──────────────┬──────────────┘
              └────────────── recuperación híbrida ┘
                      BM25 + Levenshtein + vectores
                      boosts por título y por intención
                              ▼
                  condensación de contexto (qa.rs)
                              ▼
                    contexto → modelo de lenguaje local
```

**Fuente de verdad.** El archivo `.md` en `Brain/Memory/Knowledge/bundle/<bundle>/<slug>.md` es el original.
`index.sqlite3` y el índice vectorial son derivados y descartables: `rag_rebuild_index` los reconstruye desde
los `.md`.

### Módulos

| Módulo | Líneas | Responsabilidad |
|---|---:|---|
| `lib.rs` | 1.089 | Interfaz C (FFI), comandos, registro, estado, hook previo a la consulta |
| `db.rs` | 559 | Esquema SQLite, FTS5, inserción y borrado de documentos y fragmentos |
| `search.rs` | 559 | Recuperación híbrida, fusión de puntajes, boosts, umbral de inyección |
| `qa.rs` | 477 | Condensación del contexto (similitud TF-IDF, Jaccard y Levenshtein) |
| `graph.rs` | 455 | Grafo de relaciones y mapa de similitud |
| `reindex.rs` | 345 | Reindexación y reconstrucción en segundo plano, por lotes |
| `vec_index.rs` | 312 | Índice vectorial usearch particionado |
| `frontmatter.rs` | 274 | Lectura y escritura de metadatos de los documentos |
| `embed.rs` | 182 | Vectorizador por hash y fragmentación |
| **Total** | **≈ 4.250** | |

### Decisiones de diseño

- **Ninguna llamada FFI bloquea más que unos milisegundos.** Las DLL no tienen un tiempo límite de protección
  del lado del host; por eso reindex y rebuild corren en segundo plano (`rag_job_status` informa el avance).
- **Los fallos del plugin no tumban al asistente.** Cada llamada pasa por `catch_unwind` en la frontera FFI.
- **Embeddings livianos.** Un vectorizador por hash de 512 buckets (palabras y trigramas de caracteres, CRC32,
  normalizado L2). No es un embedding semántico real: es rápido, determinista y no requiere modelo ni GPU.
- **Vectores fuera de SQLite.** Viven en `vectors.tvim` (el nombre es heredado; el archivo es 100% usearch/HNSW
  cuantizado a i8), con clave igual a `chunks.id`. Así el `.sqlite3` no duplica vectores sin cuantizar.
- **Interfaz embebida** en la DLL de release con `rust-embed`.

Constantes principales: `EMBED_DIM = 512`, `CHUNK_MAX = 900`, `CHUNK_OVERLAP = 150`, `NUM_SHARDS = 16`,
`INJECT_MIN_SCORE = 0,20`, `ALPHA = 0,5` (peso léxico/semántico), pesos de condensación 0,55 / 0,30 / 0,15.

## 4. Historia del plugin

**v1.** Con 215 mil documentos, el `plugin.py` original tardaba horas en reconstruir su índice en Python y
bloqueaba todo mientras tanto. Se reemplazó por fuerza bruta exacta en Rust (menos de 1 s contra un millón de
vectores).

**v2.** Al crecer a 4,3 millones de fragmentos, releer SQLite fila por fila al arrancar tardaba unos 13 s. Se
agregó un caché binario y carga en segundo plano.

**v3.** Bundles, enlaces explícitos y reindex/rebuild; los vectores pasaron a `sqlite-vec` (KNN contra 800 mil
vectores en unos 630 ms, sin cargar todo en memoria).

**Versión actual.** Los vectores viven en un índice usearch (HNSW) separado y cuantizado, en lugar de dentro de
SQLite. El comportamiento se mantiene: `.md` como fuente de verdad, índices reconstruibles.

### El arreglo de precisión

Con la consulta "'74 Jailbreak", la búsqueda devolvía artículos sin relación casi empatados con el correcto. La
causa: el bloque de metadatos ("Artículo de Wikipedia… Título… Tags…") casi idéntico en cada artículo caía dentro
del primer fragmento y, con un vectorizador por hash, hacía que el primer fragmento de cualquier artículo se
pareciera al de cualquier otro. La similitud entre dos artículos sin relación era 0,644 con ese bloque y 0,3776
sin él. Se corrigió en dos pasos: solo el cuerpo se fragmenta y se vectoriza (`zim_to_rag.py`), y se agregó un
**boost por título** (exacto, por solape de tokens o por subcadena) después de la fusión. Lo ya ingestado antes
del arreglo conserva el bloque y debe reimportarse o limpiarse.

## 5. Comandos del plugin

| Grupo | Comandos |
|---|---|
| Ingesta | `rag_ingest_text`, `rag_ingest_file` |
| Consulta | `rag_search` (alias `buscar_rag`), `rag_list`, `rag_get_document`, `rag_stats` |
| Edición | `rag_update_document`, `rag_delete` |
| Relaciones | `rag_okf_graph`, `rag_similarity_map` |
| Mantenimiento | `rag_reindex`, `rag_rebuild_index`, `rag_job_status`, `rag_db_stats`, `rag_wal_checkpoint`, `rag_vacuum` |
| Colecciones | `rag_list_bundles`, `rag_create_bundle`, `rag_rename_bundle`, `rag_delete_bundle` |
| Hook | `on_before_request` (inyecta contexto antes de cada consulta) |

Los comandos de búsqueda y listado aceptan `bundle` como filtro opcional (`{"bundle": "wikipedia"}` o el texto
plano).

## 6. Privacidad y normativa

- **Todo local:** ningún documento ni consulta sale del computador.
- **Supresión:** borrar un documento elimina su archivo, sus filas y sus vectores.
- **Registro:** el plugin escribe `Brain/Logs/<fecha>_plugin_rag.log` con el tiempo de cada consulta, su puntaje
  y el inicio de su texto. **Este último dato es una brecha conocida**: está planificado guardar solo la
  longitud de la consulta (tarea T12).
- **Marco legal:** Ley 19.628 vigente. La Ley 21.719 de protección de datos personales tenía vigencia fijada
  para el 1 de diciembre de 2026; el Congreso despachó el 17 de noviembre de 2026 un proyecto que la posterga al
  15 de octubre de 2027, pendiente de promulgación y publicación.

## 7. Limitaciones conocidas

Estas limitaciones están registradas como riesgos en el informe y tienen una tarea asignada:

| Limitación | Estado |
|---|---|
| **PDF:** `rag_ingest_file` rechaza PDF; solo acepta `.txt` y `.md` | Planificado en T10 |
| **DOCX:** no se extrae su contenido; se leería como texto binario | Planificado en T10 |
| **OCR** de PDF escaneados | Fuera de la v1 (cambio C-01) |
| **Búsqueda no semántica real:** el vectorizador por hash no entiende sinónimos | Aceptado; un modelo de embeddings queda para una versión posterior |
| **Validación de nombres de colección:** solo se valida al crear y renombrar, no al ingresar ni al borrar | Corrección planificada (riesgo R5) |
| **Robustez:** unos 50 `unwrap`, 24 de ellos en `lib.rs`, sobre bloqueos de mutex | Corrección planificada (riesgo R5) |
| **Pruebas:** solo 2 pruebas unitarias (en `frontmatter.rs`) | Ampliación en T13 (riesgo R6) |

## 8. Compilar y probar

Requiere Rust estable. Para Windows:

```powershell
rustup target add x86_64-pc-windows-msvc   # si no lo tienes
cargo build --release
# la DLL queda en target\release\ame_rag_plugin.dll
```

Para trabajar con datos de prueba sin tocar los documentos reales, apunta el plugin a otra carpeta:

```powershell
$env:AME_KNOWLEDGE_DIR = "C:\ruta\a\datos_de_prueba"
```

Pruebas:

```
cargo test --release
```

Si usabas el plugin Python del RAG, quítalo para que no registre los mismos nombres de comando.

## 9. Gestión del proyecto

- **Duración:** 15 semanas (semana 1: 24 de agosto de 2026). **Presupuesto:** $3.160.000, con 12% de
  contingencia; el desembolso real en dinero es de $120.000.
- **Tareas T1 a T15:** requisitos, arquitectura, formato OKF, ingesta, embeddings e índice, recuperación híbrida,
  condensación, reindexación, interfaz, soporte de PDF y DOCX, registro y set de pruebas, privacidad de logs,
  pruebas de calidad, validación piloto e informe final.
- **Ramas:** `main` (estable), `develop` (integración), `feature/<módulo>-<tarea>` y `fix/<descripción>`;
  todo cambio entra por pull request revisado por otro integrante. Commits con prefijo `feat`, `fix`, `docs`,
  `test`.

### Indicadores clave (KPI)

| KPI | Meta | Medición |
|---|---|---|
| Tasa de recuperación relevante (hit rate) | Al menos 80% | Porcentaje de preguntas de un set de prueba cuyo documento esperado queda entre los resultados |
| Latencia de recuperación | Menos de 2 s | Promedio de milisegundos que el plugin registra por consulta |
| Avance de cronograma | 100% de lo planificado | Tareas terminadas frente a las planificadas |
| Horas invertidas | 30 por semana | Bitácora individual consolidada |

Las metas de calidad (hit rate y latencia) todavía no tienen una medición real completa.

## 10. Cómo contribuir

1. Crea una rama `feature/<módulo>-<tarea>` desde `develop`.
2. Haz commits pequeños con prefijo de tipo.
3. Antes de abrir el pull request, actualiza tu rama con `develop` y ejecuta `cargo test --release`.
4. Otro integrante revisa y aprueba; un cambio de alcance se registra primero en el registro de cambios.