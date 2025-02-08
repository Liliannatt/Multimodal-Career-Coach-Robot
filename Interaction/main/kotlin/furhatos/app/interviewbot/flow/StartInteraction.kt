package furhatos.app.interviewbot.flow

import furhatos.flow.kotlin.*
import furhatos.nlu.common.No
import furhatos.nlu.common.Yes
import furhatos.app.interviewbot.flow.interview.InterviewGreeting
import furhatos.app.interviewbot.flow.Idle

val StartInteraction: State = state {
    onEntry {
        furhat.say("Hello! Would you like to participate in a job interview practice session?")
        furhat.listen()
    }

    onResponse<Yes> {
        furhat.say("Great! Let's start the interview process.")
        goto(InterviewGreeting)
    }

    onResponse<No> {
        furhat.say("No problem. Feel free to come back when you're ready for an interview practice.")
        goto(Idle)
    }

    onNoResponse {
        furhat.say("I didn't hear your response. Would you like to practice interviewing?")
        reentry()
    }
} 