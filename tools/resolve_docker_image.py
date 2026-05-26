"""
Resolve the latest Docker Hub image tag for a given repo and AMD hardware.

Usage:
  python3 resolve_docker_image.py <repo> <hardware>

Examples:
  python3 resolve_docker_image.py rocm/pytorch mi300x
  python3 resolve_docker_image.py vllm/vllm-openai-rocm mi300x
  python3 resolve_docker_image.py lmsysorg/sglang mi300x

Prints the full image:tag string to stdout on success.
Prints ERROR: <message> to stderr and exits 1 on failure.
"""

import json
import re
import sys
import urllib.request
import urllib.error

# Maps manifest hardware names to Docker Hub tag substrings.
# MI300/MI308 share the mi30x image; MI355/MI350 share mi35x.
_HW_TAG = {
    "mi300x": "mi30x",
    "mi308x": "mi30x",
    "mi355x": "mi35x",
    "mi350x": "mi35x",
}


def resolve(repo: str, hardware: str) -> str:
    """Return the full image:tag string for repo + hardware. Raises on failure."""
    org, name = repo.split("/", 1)
    url = (
        f"https://hub.docker.com/v2/repositories/{org}/{name}/tags/"
        f"?page_size=100&ordering=last_updated"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            tags = [t["name"] for t in json.loads(r.read()).get("results", [])]
    except Exception as e:
        raise RuntimeError(f"Docker Hub request failed: {e}") from e

    if repo == "vllm/vllm-openai-rocm":
        # Plain semver tags only — ROCm support is baked in, no hardware suffix
        candidates = [t for t in tags if re.match(r"^v\d+\.\d+\.\d+$", t)]
    elif repo == "rocm/pytorch":
        # Tags like rocm6.3.1_ubuntu22.04_py3.10_pytorch
        candidates = [t for t in tags if re.match(r"^rocm\d+\.\d+", t)]
    else:
        hw_suffix = _HW_TAG.get(hardware.lower(), hardware.lower())
        candidates = [t for t in tags if hw_suffix in t]

    if not candidates:
        available = ", ".join(tags[:10])
        raise RuntimeError(
            f"No tags found for {repo} matching hardware={hardware}. "
            f"Sample available tags: {available}"
        )

    def _semver(tag):
        m = re.search(r"(\d+)\.(\d+)\.?(\d*)", tag)
        return tuple(int(x) if x else 0 for x in m.groups()) if m else (0, 0, 0)

    best = max(candidates, key=_semver)
    return f"{repo}:{best}"


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <repo> <hardware>", file=sys.stderr)
        print("Example: resolve_docker_image.py rocm/pytorch mi300x", file=sys.stderr)
        sys.exit(1)

    repo, hardware = sys.argv[1], sys.argv[2]
    try:
        image = resolve(repo, hardware)
        print(image)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
