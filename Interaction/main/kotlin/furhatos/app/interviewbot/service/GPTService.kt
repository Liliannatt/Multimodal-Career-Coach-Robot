package furhatos.app.interviewbot.service

import io.ktor.client.*
import io.ktor.client.engine.apache.*
import io.ktor.client.plugins.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import org.json.JSONArray
import org.json.JSONObject
import kotlinx.coroutines.runBlocking

class GPTService(private val apiKey: String) {
    private val client = HttpClient(Apache) {
        install(HttpTimeout) {
            requestTimeoutMillis = 10000L
        }
        followRedirects = true
    }

    private val BASE_URL = "https://api.openai.com/v1/chat/completions"

    fun analyzeInterview(questions: List<String>, answers: List<String>): String = runBlocking {
        try {
            val requestBody = JSONObject().apply {
                put("model", "gpt-4o-mini")
                put("messages", JSONArray().apply {
                    put(JSONObject().apply {
                        put("role", "system")
                        put("content", "You are an expert interviewer. Analyze the interview responses and provide brief, constructive feedback in exactly 50 words.")
                    })
                    put(JSONObject().apply {
                        put("role", "user")
                        put("content", buildString {
                            appendLine("Please analyze these interview responses with a 100-word feedback:")
                            questions.zip(answers).forEach { (q, a) ->
                                appendLine("Q: $q")
                                appendLine("A: $a")
                                appendLine()
                            }
                        })
                    })
                })
                put("temperature", 0.7)
                put("max_tokens", 100)
            }

            val response = client.post(BASE_URL) {
                header("Content-Type", "application/json")
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(requestBody.toString())
            }

            val responseBody = response.bodyAsText()
            println("Response status: ${response.status}")
            println("Response body: ${responseBody.take(100)}...")

            if (response.status.isSuccess()) {
                val jsonResponse = JSONObject(responseBody)
                jsonResponse.getJSONArray("choices")
                    .getJSONObject(0)
                    .getJSONObject("message")
                    .getString("content")
            } else {
                throw Exception("API request failed: ${response.status} - $responseBody")
            }
        } catch (e: Exception) {
            println("GPT Analysis Error: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            client.close()
        }
    }
}