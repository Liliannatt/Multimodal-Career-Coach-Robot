package furhatos.app.interviewbot.flow.interview

import furhatos.flow.kotlin.*
import furhatos.nlu.common.No
import furhatos.nlu.common.Yes
import furhatos.app.interviewbot.flow.Idle
import furhatos.app.interviewbot.service.GPTService
import furhatos.app.interviewbot.util.PropertyLoader
import furhatos.records.User
import furhatos.gestures.Gestures

// List of interview questions
val interviewQuestions = listOf(
    // General interview questions
    "Could you tell me about yourself and your background?",
    "What are your greatest strengths as a developer?",
    "Where do you see yourself in five years in the tech industry?",
    "Why are you interested in joining our development team?",
    "How do you handle challenging situations in development?",
    
    // Technical CS questions
    "Describe a challenging project you've worked on and how you overcame the challenges.",
    "What technical skill have you improved the most in the last year?",
    "Share how you managed a project with a tight timeline and delivered results.",
    "What unique technical expertise can you bring to this role?",
    "How do you ensure efficient collaboration when working on code with others?"
)

// Data class to store interview responses
data class InterviewData(
    var currentQuestionIndex: Int = 0,
    val responses: MutableList<String> = mutableListOf(),
    var gptAnalysis: String = ""
)

// Extension property to store interview data for each user
var User.interviewData by NullSafeUserDataDelegate { InterviewData() }

// Initialize GPT service
private val gptService = GPTService(PropertyLoader.getProperty("openai.api.key"))

// Base state for interview interaction
val InterviewInteraction: State = state {
    onUserLeave(instant = true) {
        furhat.say("I see you're leaving. We can continue the interview when you're back.")
        goto(Idle)
    }

    onNoResponse {
        furhat.say("I didn't hear your response. Let me repeat the question.")
        reentry()
    }
}

// Initial greeting state
val InterviewGreeting: State = state(InterviewInteraction) {
    onEntry {
        // Reset interview data for new session
        users.current.interviewData = InterviewData()
        
        furhat.say {
            +"Welcome to the interview practice session."
            +"I'll be asking you some common interview questions."
            +"Are you ready to begin?"
        }
        furhat.listen()
    }

    onResponse<Yes> {
        furhat.say("Excellent! Let's start with the first question.")
        goto(StartInterview)
    }

    onResponse<No> {
        furhat.say("That's okay, take your time. Let me know when you're ready.")
        reentry()
    }
}

// Start interview state
val StartInterview: State = state(InterviewInteraction) {
    onEntry {
        val user = users.current
        if (user.interviewData.currentQuestionIndex < interviewQuestions.size) {
            if (user.interviewData.currentQuestionIndex == 5) {
                furhat.say("Now, let's move on to some technical questions.")
                delay(200)
            }
            furhat.ask(interviewQuestions[user.interviewData.currentQuestionIndex])
        } else {
            goto(EndInterview)
        }
    }

    onResponse {
        val user = users.current
        user.interviewData.responses.add(it.text)
        user.interviewData.currentQuestionIndex++

        furhat.gesture(Gestures.Smile(duration = 2.0))
        furhat.gesture(Gestures.Nod())
        
        if (user.interviewData.currentQuestionIndex == 5) {
            furhat.say("You've done well with the general questions.")
            delay(300)
        } else {
            furhat.say("Thank you for your answer.")
        }
        delay(500)

        if (user.interviewData.currentQuestionIndex < interviewQuestions.size) {
            furhat.gesture(Gestures.GazeAway())
            delay(300)
            furhat.gesture(Gestures.Smile())
            
            if (user.interviewData.currentQuestionIndex == 5) {
                furhat.say("Now, I'd like to ask you some technical questions about computer science.")
                delay(200)
            }
            
            furhat.ask(interviewQuestions[user.interviewData.currentQuestionIndex])
        } else {
            furhat.gesture(Gestures.BigSmile)
            goto(EndInterview)
        }
    }
}

// End interview state with GPT analysis
val EndInterview: State = state(InterviewInteraction) {
    onEntry {
        val user = users.current
        
        furhat.gesture(Gestures.Surprise)
        furhat.say("Thank you for completing the interview. I'm now analyzing your responses...")
        
        try {
            val analysis = gptService.analyzeInterview(
                interviewQuestions.take(user.interviewData.responses.size),
                user.interviewData.responses
            )
            user.interviewData.gptAnalysis = analysis
            
            furhat.gesture(Gestures.Smile())
            furhat.say("Here's my analysis of your interview performance:")
            
            analysis.split(". ").chunked(3).forEach { chunk ->
                furhat.say(chunk.joinToString(". "))
                delay(200)
            }
            
            furhat.gesture(Gestures.Smile(duration = 1.0))
            furhat.ask("Would you like me to repeat the analysis?")
            
        } catch (e: Exception) {
            furhat.gesture(Gestures.ExpressSad)
            furhat.say("I apologize, but I encountered an error while generating the analysis. " +
                      "However, I can tell you completed the interview well.")
            goto(Idle)
        }
    }
    
    onResponse<Yes> {
        val user = users.current
        furhat.gesture(Gestures.Smile())
        furhat.say(user.interviewData.gptAnalysis)
        furhat.gesture(Gestures.Smile(duration = 2.0))
        furhat.say("I hope this feedback was helpful. Good luck with your future interviews!")
        goto(Idle)
    }
    
    onResponse<No> {
        furhat.gesture(Gestures.Smile(duration = 2.0))
        furhat.say("I hope this feedback was helpful. Good luck with your future interviews!")
        goto(Idle)
    }
} 