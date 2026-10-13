#!/usr/bin/env python3
"""Extract the ROOT-resident part of fillTrackAnalysisTree preliminary cuts.

Event IDs from root2h5_reco.py encode the ROOT filename and an entry number
after its optional dummy first entry. The unavailable BRecoMuonBDTEval
variables are intentionally not approximated here.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import uproot

BRANCHES = {
    "fNHits": "BRecoMuon./BRecoMuon.fNHits",
    "fNStrings": "BRecoMuon./BRecoMuon.fNStrings",
    "fCovMatrStat": "BRecoMuon./BRecoMuon.fCovMatrixStatus",
    "fScfMaxTheta": "BRecoMuon./BRecoMuon.fScfMaxTheta",
    "fEvCenterZ": "BRecoMuon./BRecoMuon.fEvCenterZ",
    "fZDist": "BRecoMuon./BRecoMuon.fZDist",
    "fTimeTerm": "BRecoMuon./BRecoMuon.fTimeChi2",
    "fNTriplets": "BRecoMuon./BRecoMuon.fNTriplets",
    "fThetaErr": "BRecoMuon./BRecoMuon.fThetaErr",
    "fNCalls": "BRecoMuon./BRecoMuon.fNCalls",
    "fThetaRec": "BRecoMuon./BRecoMuon.fThetaRec",
    "fPhiRec": "BRecoMuon./BRecoMuon.fPhiRec",
}
PULSE_COUNT = "BEvent./BEvent.fPulseN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-ids", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    by_file: dict[str, list[tuple[str, int]]] = defaultdict(list)
    ids = [line.strip() for line in args.event_ids.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event IDs")
    for event_id in ids:
        if not event_id.startswith("nue2_"):
            raise ValueError(f"Unexpected particle in {event_id}")
        stem, index = event_id.removeprefix("nue2_").rsplit("_", 1)
        by_file[stem].append((event_id, int(index)))
    stems = sorted(by_file)
    if args.max_files is not None:
        stems = stems[: args.max_files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.csv")
    fieldnames = ["event_id", "root_entry", "dummy_entry_offset", *BRANCHES]
    written = 0
    try:
        with partial.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for number, stem in enumerate(stems, 1):
                root_path = args.root_dir / f"{stem}.root"
                if not root_path.is_file():
                    raise FileNotFoundError(root_path)
                with uproot.open(root_path) as root:
                    events = root["Events"]
                    first_pulse_count = events[PULSE_COUNT].array(
                        entry_start=0, entry_stop=1, library="np"
                    )[0]
                    offset = int(first_pulse_count == 0)
                    arrays = {
                        name: events[branch].array(library="np")
                        for name, branch in BRANCHES.items()
                    }
                    for event_id, source_index in by_file[stem]:
                        entry = source_index + offset
                        if not 0 <= entry < events.num_entries:
                            raise IndexError(f"{event_id}: ROOT entry {entry} out of range")
                        row = {
                            "event_id": event_id,
                            "root_entry": entry,
                            "dummy_entry_offset": offset,
                        }
                        row.update({name: float(array[entry]) for name, array in arrays.items()})
                        writer.writerow(row)
                        written += 1
                if number % 25 == 0:
                    print(f"files={number}/{len(stems)} events={written}", flush=True)
        partial.replace(args.output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"complete files={len(stems)} events={written} output={args.output}")


if __name__ == "__main__":
    main()
