---
license: cc-by-nc-sa-4.0
pretty_name: ATP data for the tennis-hof project (copy of Jeff Sackmann's data)
tags:
  - tennis
  - sports
  - atp
language:
  - en
configs:
  - config_name: matches
    data_files: "atp/atp_matches_[0-9]*.csv"
  - config_name: rankings
    data_files: "atp/atp_rankings_*.csv"
  - config_name: players
    data_files: "atp/atp_players.csv"
---

# ATP data for the tennis-hof project

The pinned data source for [jmatarese/tennis-hof](https://github.com/jmatarese/tennis-hof),
an objective ranking of ATP players for the International Tennis Hall of Fame.

All data was collected and compiled by **Jeff Sackmann**. This is a copy kept by
the project so the version it uses can't change underneath it.

## Where it came from

Duplicated on 2026-10-21 from
[Aneeshers/tennis-sackmann-archive](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive)
at commit `8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993`, an archival mirror of:

- ATP: https://github.com/JeffSackmann/tennis_atp
- WTA: https://github.com/JeffSackmann/tennis_wta
- Grand Slam point-by-point: https://github.com/JeffSackmann/tennis_slam_pointbypoint

Each folder keeps Sackmann's original README as `UPSTREAM_README.md`.

## What the project uses

| Viewer table | Files | What it is |
|---|---|---|
| `matches` | `atp/atp_matches_YYYY.csv` | Tour-level singles matches, 1968 on, one file per year |
| `rankings` | `atp/atp_rankings_*.csv` | Weekly ATP rankings (the rankings start in 1973) |
| `players` | `atp/atp_players.csv` | One row per player: id, name, hand, birth date, country |

Column definitions: `atp/matches_data_dictionary.txt`. The `wta/` and
`slam_pointbypoint/` folders, and the challenger, futures and qualifying
files in `atp/`, are kept but not currently used.

## Versions

Each version is a tag. The project pins the tag's **commit hash** in
`data/snapshot.toml`, because a tag can be moved and a hash can't.

| Tag | What changed |
|---|---|
| `v1` | The source as duplicated, plus this card |

## License

**CC BY-NC-SA 4.0**, Jeff Sackmann's original license. Credit Jeff Sackmann
and link the original repositories; non-commercial use only; anything derived
from this data must be shared under the same license. This copy claims no
ownership over the data and adds no rights to it.
