# Python environment for the public OAuth front.
{ pkgs }:

let
  # mcp pinned to 1.x so nixpkgs' fastmcp 3.x isn't broken; see ./mcp-1.nix.
  python = pkgs.python3.override {
    self = python;
    packageOverrides = pyfinal: _: {
      mcp = pyfinal.callPackage ./mcp-1.nix { };
    };
  };
in
(python.withPackages (ps: [
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
