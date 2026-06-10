package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateListen(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		addr             string
		allowNonLoopback bool
		ok               bool
	}{
		"loopback ip":         {"127.0.0.1:8484", false, true},
		"ipv6 loopback":       {"[::1]:8484", false, true},
		"localhost":           {"localhost:8484", false, true},
		"all interfaces":      {"0.0.0.0:8484", false, false},
		"lan ip":              {"192.168.1.10:8484", false, false},
		"all interfaces ok'd": {"0.0.0.0:8484", true, true},
		"missing port":        {"127.0.0.1", false, false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			err := validateListen(tc.addr, tc.allowNonLoopback)
			if tc.ok {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
			}
		})
	}
}

func TestToolPolicyDefaultsRestrictive(t *testing.T) {
	t.Parallel()

	allow, disallow := toolPolicy("", "")
	require.Empty(t, allow)
	require.Subset(t, disallow, []string{"Bash", "Write", "Edit"},
		"with no operator policy, dangerous built-ins must be disallowed by default")

	allow, disallow = toolPolicy("Read,Glob", "")
	require.Equal(t, []string{"Read", "Glob"}, allow)
	require.Empty(t, disallow, "an explicit allowlist suppresses the restrictive default")

	allow, disallow = toolPolicy("", "Bash")
	require.Empty(t, allow)
	require.Equal(t, []string{"Bash"}, disallow, "an explicit disallowlist is honored verbatim")
}
