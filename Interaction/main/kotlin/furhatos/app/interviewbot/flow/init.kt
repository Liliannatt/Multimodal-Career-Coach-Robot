package furhatos.app.interviewbot.flow

import furhatos.flow.kotlin.*
import furhatos.skills.SimpleEngagementPolicy
import furhatos.skills.SingleUserEngagementPolicy
import furhatos.app.interviewbot.flow.StartInteraction
import furhatos.app.interviewbot.flow.Idle

val Init: State = state {
    onEntry {
        /** Set our default interaction parameters */
        users.setSimpleEngagementPolicy(distance = 1.5, maxUsers = 2)
        // furhat.character = "Marty"
        // furhat.voice = ""
        when {
            users.hasAny() -> {
                furhat.attend(users.random)
                goto(StartInteraction)
            }
            else -> goto(Idle)
        }
    }
}