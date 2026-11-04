# Local review and report gates

The analytics path requires an explicit confirmed local association with matching run ID, local track ID, positive revision and named reviewer. A player label, tracker state, calibration or generated track-reference string cannot establish review. Ambiguous observations remain excluded from physical estimates even when an identity is reviewed.

Two-run comparison reads each run's associations.json and requires its current record to match the analytics identity/revision snapshot and requested player. Different display labels do not create two sessions: equal run IDs, session IDs, known source references or source hashes block comparison. Legacy analytics without reviewed records must be regenerated from a reviewed ledger; missing fields are not filled from request parameters.

Report hashes bind schema, report ID, revision, player, lineage, objective, claims, coach notes and recipient scope. Approval recomputes the current hash. Parent/guardian reads recompute it again and require the approved hash and revision to match. Changes require a new reviewed draft; old approval strings alone cannot authorize changed content.

These are local artifact-consistency checks. A ledger file, reviewer string or hash is not server authentication, an independent measurement or proof of review by an authorized coach. A real service still needs authenticated actors, protected persistence, school/player permissions, current association/revocation checks and independent field evidence. No physical/model/guardian acceptance is inferred from synthetic unit tests.

The user forbids agent access to the real .env and credential files. Development verification uses synthetic records and mocked providers only. Operator-configured credentials remain outside code, reports, model prompts and test output.
