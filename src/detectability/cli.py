"""Command-line interface: ``detectability {demo,train,decay,predict,serve}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data as data_mod
from .decay import plot_decay, run_decay
from .evaluate import compare_detectors
from .models import DETECTORS, ModelBundle, fit_bundle
from .transforms import TRANSFORMS, TransformationSchedule


def _print_table(df) -> None:
    print(df.round(4).to_string())


def cmd_demo(args: argparse.Namespace) -> int:
    from .synthetic import generate_corpus

    df = generate_corpus(n_per_class=300, seed=args.seed)
    splits = data_mod.split(df, seed=args.seed)
    table, bundles = compare_detectors(splits.train, splits.test, ["random_forest"], "synthetic demo corpus", args.seed)
    path = bundles["random_forest"].save(Path(args.out) / "random_forest.joblib")
    _print_table(table[["accuracy", "precision", "recall", "f1"]])
    print(f"\nSaved demo model to {path}. Start the site with: detectability serve")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    df = data_mod.load_dataset(args.data, max_samples=args.max_samples, seed=args.seed)
    print(f"Loaded {len(df):,} texts {data_mod.class_balance(df['label'])} from {args.data}")
    splits = data_mod.split(df, test_size=args.test_size, seed=args.seed)
    names = list(DETECTORS) if args.compare else [args.detector]
    table, bundles = compare_detectors(splits.train, splits.test, names, Path(args.data).name, args.seed)
    _print_table(table[["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]])

    out = Path(args.out)
    for name, bundle in bundles.items():
        bundle.save(out / f"{name}.joblib")
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "detector_comparison.csv")
    print(f"\nModels saved to {out}/, metrics to {results / 'detector_comparison.csv'}")
    return 0


def cmd_decay(args: argparse.Namespace) -> int:
    df = data_mod.load_dataset(args.data, max_samples=args.max_samples, seed=args.seed)
    splits = data_mod.split(df, test_size=args.test_size, seed=args.seed)
    print(f"Training {len(args.detectors)} detectors on {len(splits.train):,} texts...")
    bundles = {
        name: fit_bundle(
            name, splits.train["text"].tolist(), splits.train["label"].to_numpy(), Path(args.data).name, args.seed
        )
        for name in args.detectors
    }
    ai_texts = splits.test.loc[splits.test["label"] == 1, "text"].tolist()
    if args.max_ai_texts:
        ai_texts = ai_texts[: args.max_ai_texts]
    schedule = TransformationSchedule(
        names=args.transforms, per_step=args.per_step, strength=args.strength, seed=args.seed
    )
    print(f"Transforming {len(ai_texts):,} held-out AI texts over {args.generations} generations...")
    result = run_decay(bundles, ai_texts, schedule, generations=args.generations)
    out = result.save(args.out)
    plot_decay(result, out / "decay_curves.png")
    print()
    print(result.summary_markdown())
    print(f"Results written to {out}/")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundle = ModelBundle.load(args.model)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        print("Provide text or --file", file=sys.stderr)
        return 2
    p_ai = float(bundle.predict_proba([text])[0])
    print(json.dumps({"p_ai": round(p_ai, 4), "label": "ai" if p_ai >= 0.5 else "human"}))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(model_path=args.model), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detectability", description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="train a demo model on the synthetic corpus (no download)")
    p.add_argument("--out", default="models")
    p.set_defaults(func=cmd_demo)

    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", required=True, help="HC3 jsonl, or CSV/JSONL with text,label columns")
        p.add_argument("--max-samples", type=int, default=None, help="stratified subsample size")
        p.add_argument("--test-size", type=float, default=0.2)

    p = sub.add_parser("train", help="train and evaluate detector(s)")
    add_data_args(p)
    p.add_argument("--detector", choices=sorted(DETECTORS), default="random_forest")
    p.add_argument("--compare", action="store_true", help="train and compare every detector")
    p.add_argument("--out", default="models")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("decay", help="run the successive-transformation decay experiment")
    add_data_args(p)
    p.add_argument("--detectors", nargs="+", choices=sorted(DETECTORS), default=list(DETECTORS))
    p.add_argument("--transforms", nargs="+", choices=sorted(TRANSFORMS), default=list(TRANSFORMS))
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--per-step", type=int, default=2)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--max-ai-texts", type=int, default=None)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_decay)

    p = sub.add_parser("predict", help="score a single text")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--file")
    p.add_argument("--model", default="models/random_forest.joblib")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--model", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
