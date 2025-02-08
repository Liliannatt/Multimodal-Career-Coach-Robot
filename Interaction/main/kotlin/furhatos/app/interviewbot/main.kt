package furhatos.app.interviewbot

import furhatos.app.interviewbot.flow.Init
import furhatos.skills.Skill
import furhatos.flow.kotlin.*

class InterviewbotSkill : Skill() {
    override fun start() {
        Flow().run(Init)
    }
}

fun main(args: Array<String>) {
    Skill.main(args)
}


