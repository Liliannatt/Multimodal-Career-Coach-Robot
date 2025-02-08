package furhatos.app.interviewbot.util

import java.util.Properties

object PropertyLoader {
    private val properties = Properties().apply {
        PropertyLoader::class.java.getResourceAsStream("/application.properties")?.use {
            load(it)
        } ?: println("Warning: application.properties not found")
    }

    fun getProperty(key: String): String {
        val value = properties.getProperty(key)?.trim()?.replace("\"", "")
            ?: throw IllegalStateException("Property $key not found")
        println("Loaded property $key: ${value.take(10)}...")
        return value
    }
} 