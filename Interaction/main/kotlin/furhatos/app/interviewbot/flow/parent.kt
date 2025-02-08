package furhatos.app.interviewbot.flow

import furhatos.flow.kotlin.*
import furhatos.nlu.common.RequestRepeat
import furhatos.records.User

/**
 * Parent state for all interview interactions
 * Contains common behaviors and error handling
 */
val Parent: State = state {
    // Store last state for each user
    var lastState: State? = null
    
    // Handle user leaving
    onUserLeave(instant = true) {
        lastState = currentState
        if (users.count > 0) {
            furhat.attend(users.other)
            goto(StartInteraction)
        } else {
            goto(Idle)
        }
    }

    // Handle new user entering
    onUserEnter(instant = true) {
        if (lastState != null) {
            furhat.say("Would you like to continue where we left off?")
            furhat.listen()
        } else {
            furhat.glance(it)
        }
    }

    // Handle request to repeat
    onResponse<RequestRepeat> {
        furhat.say("Let me repeat that.")
        reentry()
    }

    // Handle no response
    onNoResponse {
        furhat.say("I didn't hear anything. Could you please respond?")
        reentry()
    }

    // Handle unrecognized responses
    onResponse {
        furhat.say("I'm sorry, I didn't quite understand that. Could you please rephrase?")
        reentry()
    }
} 