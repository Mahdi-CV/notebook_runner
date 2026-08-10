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

    hw_suffix = _HW_TAG.get(hardware.lower(), hardware.lower())

    if repo == "rocm/pytorch":
        # Tags look like: rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
        # Filter for release tags (not nightly/internal), prefer latest ROCm + Python version.
        candidates = [
            t for t in tags
            if "pytorch_release" in t or "pytorch_release" in t.replace("-", "_")
        ]
        if not candidates:
            # Fallback: any tag with a ROCm version string
            candidates = [t for t in tags if re.search(r"rocm\d+\.\d+", t)]

        if candidates:
            def _pytorch_sort_key(tag):
                rocm_m = re.search(r"rocm(\d+)\.(\d+)\.?(\d*)", tag)
                rocm_ver = tuple(int(x) if x else 0 for x in rocm_m.groups()) if rocm_m else (0, 0, 0)
                pt_m = re.search(r"pytorch[_-]release[_-](\d+)\.(\d+)\.?(\d*)", tag)
                pt_ver = tuple(int(x) if x else 0 for x in pt_m.groups()) if pt_m else (0, 0, 0)
                py312 = 1 if "py3.12" in tag else 0
                ubuntu24 = 1 if "ubuntu24" in tag else 0
                return (rocm_ver, pt_ver, py312, ubuntu24)
            best = max(candidates, key=_pytorch_sort_key)
            return f"{repo}:{best}"

    elif repo == "vllm/vllm-openai-rocm":
        # Filter out 'latest' and nightly tags; prefer versioned release tags
        candidates = [
            t for t in tags
            if t != "latest"
            and not t.startswith("nightly")
            and not t.endswith("-base")
            and re.search(r"\d+\.\d+", t)
        ]
        if candidates:
            def _vllm_sort_key(tag):
                # Extract primary version (e.g., v0.8.3 or 0.8.3)
                ver_m = re.match(r"v?(\d+)\.(\d+)\.?(\d*)", tag)
                ver = tuple(int(x) if x else 0 for x in ver_m.groups()) if ver_m else (0, 0, 0)
                rocm_m = re.search(r"rocm(\d+)\.(\d+)\.?(\d*)", tag)
                rocm_ver = tuple(int(x) if x else 0 for x in rocm_m.groups()) if rocm_m else (0, 0, 0)
                return (ver, rocm_ver)
            best = max(candidates, key=_vllm_sort_key)
            return f"{repo}:{best}"

    elif repo == "rocm/dgl":
        candidates = [t for t in tags if re.match(r"^dgl-\d+\.\d+", t)]
        def _dgl_sort_key(tag):
            m = re.search(r"rocm(\d+)\.(\d+)\.?(\d*)", tag)
            rocm_ver = tuple(int(x) if x else 0 for x in m.groups()) if m else (0, 0, 0)
            py312 = 1 if "py3.12" in tag else 0
            ubuntu24 = 1 if "ubuntu24" in tag else 0
            return (rocm_ver, py312, ubuntu24)
        best = max(candidates, key=_dgl_sort_key)
        return f"{repo}:{best}"
    elif repo == "lmsysorg/sglang":
        candidates = [t for t in tags if hw_suffix in t]
    else:
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
