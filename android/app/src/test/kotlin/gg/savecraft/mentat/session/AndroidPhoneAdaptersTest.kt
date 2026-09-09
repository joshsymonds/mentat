package gg.savecraft.mentat.session

import android.Manifest
import android.app.Application
import android.content.ContentProvider
import android.content.ContentValues
import android.content.Intent
import android.content.pm.ProviderInfo
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.Bundle
import android.provider.ContactsContract
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.ContactMatch
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowContentResolver
import org.robolectric.shadows.ShadowSettings

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class AndroidPhoneAdaptersTest {
    private lateinit var application: Application

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        Shadows.shadowOf(application).clearRegisteredReceivers()
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.READ_CONTACTS)
        ShadowSettings.setCanDrawOverlays(false)
    }

    @Test
    fun intentLauncherStartsViewIntentWithUriAndNewTask() {
        AndroidIntentLauncher(application).launch("https://example.test/x")

        val started = Shadows.shadowOf(application).nextStartedActivity
        assertNotNull(started)
        assertEquals(Intent.ACTION_VIEW, started.action)
        assertEquals("https://example.test/x", started.dataString)
        assertTrue(started.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)
    }

    @Test
    fun intentLauncherReflectsOverlayPermission() {
        ShadowSettings.setCanDrawOverlays(false)
        assertTrue(!AndroidIntentLauncher(application).canDrawOverlays())

        ShadowSettings.setCanDrawOverlays(true)
        assertTrue(AndroidIntentLauncher(application).canDrawOverlays())
    }

    @Test
    fun contactResolverUsesSingleEncodedFilterPathAndReturnsRows() {
        val provider = RecordingContactsProvider()
        ShadowContentResolver.registerProviderInternal(ContactsContract.AUTHORITY, provider)

        val matches = AndroidContactResolver(application).resolve("Sarah Jane")

        assertEquals("Sarah Jane", provider.queriedUri?.lastPathSegment)
        assertEquals(
            listOf(
                ContactMatch("Sarah Jane", "+15551212"),
                ContactMatch("Sarah Jane Work", "+15559876"),
            ),
            matches,
        )
    }

    private class RecordingContactsProvider : ContentProvider() {
        var queriedUri: Uri? = null

        override fun onCreate(): Boolean = true

        override fun query(
            uri: Uri,
            projection: Array<out String>?,
            selection: String?,
            selectionArgs: Array<out String>?,
            sortOrder: String?,
        ): Cursor {
            queriedUri = uri
            return MatrixCursor(
                arrayOf(
                    ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                    ContactsContract.CommonDataKinds.Phone.NUMBER,
                ),
            ).apply {
                addRow(arrayOf("Sarah Jane", "+15551212"))
                addRow(arrayOf("Sarah Jane Work", "+15559876"))
            }
        }

        override fun getType(uri: Uri): String? = null

        override fun insert(uri: Uri, values: ContentValues?): Uri? = null

        override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0

        override fun update(
            uri: Uri,
            values: ContentValues?,
            selection: String?,
            selectionArgs: Array<out String>?,
        ): Int = 0
    }
}
