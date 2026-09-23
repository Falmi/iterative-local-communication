# Security and privacy audit — local draft

Scope: curated publishable files only; ignored outputs/caches are not release content.

| Category | Outcome |
|---|---|
| Credential patterns, private-key headers, access-token formats | No matches |
| Authentication-related terms requiring review | No matches in candidate files |
| Personal home paths and Windows user paths | No matches after path sanitization |
| Email addresses | No matches |
| Environment files, key files, cookies | Excluded; none copied |
| Dataset image files and large model checkpoints | Excluded |
| Unrelated manuscripts/projects | Not selected |
| Internal/private URLs and student records | None identified in selected materials |
| Git LFS dependency | None configured |
| Git history credentials | Initial staged tree checked; no credential matches |

The audit records categories only, never secret values. Automated pattern checks cannot prove the absence of every possible secret. The added historical snapshot passed credential/home-path pattern checks and all 38 original source hashes. Historical version mapping is documented separately. Publication has not occurred.
