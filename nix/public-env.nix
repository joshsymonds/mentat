# Python environment for the public OAuth front.
{ pkgs }:

(pkgs.python3.withPackages (ps: [
  ps.fastmcp
  ps.uvicorn
])).overrideAttrs (old: {
  # Keep the package import checks inside the derivation so deployment cannot
  # accidentally select an environment without the FastMCP auth/proxy APIs.
  postBuild = (old.postBuild or "") + ''
    $out/bin/python - <<'PY'
    import fastmcp.server.auth.oidc_proxy
    import fastmcp.server.providers.proxy
    import uvicorn
    PY
  '';
})
