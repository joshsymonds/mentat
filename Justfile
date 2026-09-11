# lint + test (default)
default: lint test test-ha test-voice test-public

# eslint (strict-type-checked) + tsc --strict + knip
lint:
    npm run lint

# vitest, offline (no claude binary, no network)
test:
    npm test

# HA custom component (ha/): stdlib-only, no homeassistant install needed
test-ha:
    python3 -m unittest discover -s ha/tests

# Voice stream adapter (voice/): stdlib-only, no livekit install needed
test-voice:
    python3 -m unittest discover -s voice/tests

# Public OAuth front (public/): FastMCP and Uvicorn from the flake environment
test-public:
    "$(nix build .#public-env --no-link --print-out-paths)/bin/python" -m unittest discover -s public/tests

# Create a new worktree at .worktrees/BRANCH.
new-worktree BRANCH:
    #!/usr/bin/env bash
    set -euo pipefail
    # just's quote() wraps the argument in a shell single-quote literal
    # (escaping any interior quote) before bash parses this line, so a value
    # like $(cmd) or `cmd` is never command-substituted. Interpolating the
    # parameter bare inside double quotes would leave that injection open.
    branch={{quote(BRANCH)}}
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        git worktree add ".worktrees/${branch}" "${branch}"
    else
        git worktree add -b "${branch}" ".worktrees/${branch}"
    fi
    echo ""
    echo "Worktree ready: .worktrees/${branch}"
    echo "Next: cd .worktrees/${branch}"

# Remove the worktree at .worktrees/BRANCH.
rm-worktree BRANCH:
    branch={{quote(BRANCH)}}; git worktree remove ".worktrees/${branch}"

# Android Lint + Gradle tests in the lean SDK shell (no emulator, no image).
test-android:
    nix develop .#android-unit -c bash -ceu 'cd android && gradle lintDebug test'

# Pair the phone first: `adb pair <host>:<port>`, then `adb connect <host>:<port>`.
# Build the debug APK and install it on the default adb device.
android-install:
    nix develop .#android-unit -c bash -ceu '\
        (cd android && gradle assembleDebug); \
        adb install -r android/app/build/outputs/apk/debug/app-debug.apk'

# Boot a headless emulator and prove the assistant role receives KEYCODE_ASSIST.
android-e2e:
    nix develop .#android -c bash -ceu '\
        export ANDROID_AVD_HOME="$PWD/android/.avd"; \
        export ANDROID_USER_HOME="$PWD/android/.android"; \
        export GRADLE_USER_HOME="$PWD/android/.gradle"; \
        trap "android/scripts/emulator.sh stop; adb kill-server" EXIT; \
        (cd android && gradle assembleDebug); \
        android/scripts/emulator.sh create-avd; \
        serial="$(android/scripts/emulator.sh start)"; \
        adb -s "$serial" install -r android/app/build/outputs/apk/debug/app-debug.apk; \
        api_level="$(adb -s "$serial" shell getprop ro.build.version.sdk)"; \
        printf "Assistant role image: google_apis API %s\\n" "$api_level"; \
        adb -s "$serial" shell cmd role add-role-holder --user 0 android.app.role.ASSISTANT gg.savecraft.mentat; \
        adb -s "$serial" logcat -c; \
        adb -s "$serial" shell input keyevent KEYCODE_ASSIST; \
        for _ in $(seq 1 30); do \
          if adb -s "$serial" logcat -d -s MentatAssist:I | grep -Fq MENTAT_ASSIST_RECEIVED; then \
            adb -s "$serial" logcat -d -s MentatAssist:I | grep -F MENTAT_ASSIST_RECEIVED; \
            exit 0; \
          fi; \
          sleep 1; \
        done; \
        adb -s "$serial" logcat -d -s MentatAssist:I; \
        exit 1'

# Boot a headless emulator and run the real-SFU Android instrumentation lane.
android-live ENDPOINT:
    nix develop .#android -c bash -ceu '\
        endpoint="$1"; \
        export ANDROID_AVD_HOME="$PWD/android/.avd"; \
        export ANDROID_USER_HOME="$PWD/android/.android"; \
        export GRADLE_USER_HOME="$PWD/android/.gradle"; \
        trap "android/scripts/emulator.sh stop; adb kill-server" EXIT; \
        (cd android && gradle assembleDebug assembleDebugAndroidTest); \
        android/scripts/emulator.sh create-avd; \
        serial="$(android/scripts/emulator.sh start)"; \
        (cd android && gradle connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.mentatTokenEndpoint="$endpoint")' -- {{quote(ENDPOINT)}}
