package furhatos.app.interviewbot.flow

import furhatos.flow.kotlin.*

val Idle: State = state {
    onEntry {
        furhat.attendNobody()
    }

    onUserEnter {
        furhat.attend(it)
        goto(StartInteraction)
    }
} 