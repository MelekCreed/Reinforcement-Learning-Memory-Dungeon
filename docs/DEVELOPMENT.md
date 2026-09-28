# Development record

The project was built from an empty workspace in stages: environment and oracle validation; feed-forward/history/recurrent policies and sequence PPO; cloned-state interventions; formal experiment scripts; dashboard; measured results and documentation.

The global instructions referenced `~/.Codex/templates/AI_TEAM.md` and `AGENTS.md`. Both template files were absent. No replacement collaboration protocol was invented and no auxiliary agent was used. Local Git commits record the implementation stages.

Environment validation uses a privileged shortest-path oracle only in tests. Learned policies never receive the oracle, the complete map, observation truth labels, or the symbolic trace.

The first dashboard test found an invalid Streamlit icon; it was removed. CPU experiment jobs can contend with Streamlit startup, so its integration test has a longer timeout than the numerical unit tests.

Checkpoints stay local and are excluded from Git. Recorded metrics, configurations, paired episode evidence and plots are versioned. The repository is created privately by default; publication visibility can be changed explicitly by its owner.
