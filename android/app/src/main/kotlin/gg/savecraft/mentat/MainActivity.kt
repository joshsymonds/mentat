package gg.savecraft.mentat

/**
 * The home-screen and "open Mentat" entry point. Assist dispatch reaches [AssistActivity]
 * through a signature-guarded component that only the system can start; a launcher entry
 * has to be startable by the launcher and by whichever assistant the user asks to open
 * the app, so it is a separate, unguarded component running the same session.
 */
class MainActivity : AssistActivity()
