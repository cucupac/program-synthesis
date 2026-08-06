"""Generate starter, validation, and test problem sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiment.dsl import render_grid
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_sweep, world_to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(str(config_path))
    output_dir = Path(config.output_dir)
    worlds_dir = output_dir / "worlds"
    shutil.rmtree(worlds_dir, ignore_errors=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, output_dir / "generator.yaml")

    worlds = make_sweep(config)
    world_files = []
    for world in worlds:
        filename = f"seed_{world.world_seed}__{world.condition.name}.json"
        world_path = worlds_dir / filename
        with open(world_path, "w", encoding="utf-8") as file:
            json.dump(world_to_dict(world), file, indent=2, sort_keys=True)
            file.write("\n")
        world_files.append(str(world_path))

    metadata = {
        "config": str(output_dir / "generator.yaml"),
        "world_count": len(worlds),
        "world_files": world_files,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    _write_preview(output_dir / "preview.txt", worlds)


def _write_preview(path: Path, worlds) -> None:
    lines = []
    for world in worlds[:2]:
        lines.append(f"world seed={world.world_seed} condition={world.condition.name}")
        for task in world.tasks_start[:3]:
            lines.append(f"{task.id} motifs={','.join(task.motif_ids)} glue={','.join(task.glue_ops) or 'none'}")
            lines.append(render_grid(task.target))
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
