import enum
from typing import List, Optional

from pydantic import BaseModel


class SkillName(enum.Enum):
    LOGIC = "Logic"
    ENCYCLOPEDIA = "Encyclopedia"
    RHETORIC = "Rhetoric"
    DRAMA = "Drama"
    ELECTROCHEMISTRY = "Electrochemistry"
    SHIVERS = "Shivers"
    HALF_LIGHT = "Half Light"
    COMPOSURE = "Composure"
    HORRIFIC_NECKTIE = "Horrific Necktie"
    INTERFACING = "Interfacing"
    SAVOIR_FAIRE = "Savoir Faire"
    REACTION_SPEED = "Reaction Speed"
    PERCEPTION = "Perception"
    HAND_EYE_COORDINATION = "Hand/Eye Coordination"
    ENDURANCE = "Endurance"
    SUGGESTION = "Suggestion"
    EMPATHY = "Empathy"
    INLAND_EMPIRE = "Inland Empire"
    CONCEPTUALIZATION = "Conceptualization"
    VISUAL_CALCULUS = "Visual Calculus"
    VOLITION = "Volition"
    AUTHORITY = "Authority"
    ESPRIT_DE_CORPS = "Esprit de Corps"
    PAIN_THRESHOLD = "Pain Threshold"
    PHYSICAL_INSTRUMENT = "Physical Instrument"


class SkillCategory(enum.Enum):
    INTELLECT = "Intellect"
    PSYCHE = "Psyche"
    PHYSIQUE = "Physique"
    MOTORICS = "Motorics"


class Skill(BaseModel):
    name: SkillName
    category: Optional[SkillCategory]
    description: str
    quotes: List[str]

    @classmethod
    def from_name(cls, name: SkillName) -> "Skill":
        for skill in skills:
            if skill.name == name:
                return skill
        raise ValueError("Skill not found")


skills: List[Skill] = [
    Skill(
        name=SkillName.LOGIC,
        category=SkillCategory.INTELLECT,
        description="Wield raw intellectual power. Deduce the world.",
        quotes=[
            "Why? There is still a way to win her back, you know. All you need to do is *analyze*.",
            "Do it for the picture puzzle. Put it all together. Solve the world. One conversation at a time.",
            "51 - 8 = 43",
            "No objections. It's mathematically impossible to achieve a classless society. Everyone knows this.",
        ],
    ),
    Skill(
        name=SkillName.ENCYCLOPEDIA,
        category=SkillCategory.INTELLECT,
        description="Call upon all your knowledge. Produce fascinating trivia.",
        quotes=[
            "Your mangled brain would like you to know there is a boxer called Contact Mike.",
            "You see noble aims but also mountains upon mountains of corpses. They should've taken a more *measured* approach. Also, you're really smart.",
            "Let her *go?* This is the holy queen of the territories of Mundi and Insulinde! Think of the historic knowledge we could glean! This is a once-in-a-lifetime opportunity -- to win her back!",
        ],
    ),
    Skill(
        name=SkillName.RHETORIC,
        category=SkillCategory.INTELLECT,
        description="Practice the art of persuasion. Enjoy rigorous intellectual discourse.",
        quotes=[
            "Hear that? That's the sound of meaninglessness. Meaning, ideas, theory -- all that has evaporated. Now there is only dry silence -- the sound of a mind made up. Just like four billion others. I am so sorry.",
            "Say one of these fascist or communist things or fuck off.",
            "Not at all. Complaining about other communists is one of the most important parts of being a communist.",
            "All this eloquence — it's in service of something. She's afraid.",
        ],
    ),
    Skill(
        name=SkillName.DRAMA,
        category=SkillCategory.INTELLECT,
        description="Play the actor. Lie and detect lies.",
        quotes=[
            "Lies, sire! She cannot but love you. She has said so a hundred times.",
            "She thinks you are an idiot, sire.",
            "How dull! Entertainment isn't a wall to read. Where are the performers and pyrotechnics? No one will pay attention to this.",
        ],
    ),
    Skill(
        name=SkillName.CONCEPTUALIZATION,
        category=SkillCategory.INTELLECT,
        description="Understand creativity. See Art in the world.",
        quotes=[
            "I'm sorry. I'm foam. All I can do is foam, it's meaningless.",
            "Of course not, it's autism. Box-drawing. Masturbation with a ruler and a sextant or whatever they use.",
            "It's death — but for the Universe? Oh, we're contemplating the living *shit* out of this.",
            "There is definitely something futuristic about his hair, aggressively so. You get the sense that *this* is what the future will look like...",
        ],
    ),
    Skill(
        name=SkillName.VISUAL_CALCULUS,
        category=SkillCategory.INTELLECT,
        description="Reconstruct crime scenes. Make laws of physics work for the Law.",
        quotes=[
            "Track his gaze. He's looking out past the broken wall, toward the opposite side of the Bay...",
            "The man does not know the bullet has entered his brain. He never will. Death comes faster than the realization.",
            "The sun blazes high up in the sky... baking the planks, the sand, your skin. The order was carried out in the afternoon.",
        ],
    ),
    Skill(
        name=SkillName.VOLITION,
        category=SkillCategory.PSYCHE,
        description="Hold yourself together. Keep your Morale up.",
        quotes=[
            "In honour of your shit, lieutenant-yefreitor. Which you kept *together* in the face of total, unrelenting terror. Day after day. Second by second.",
            "I can't help you. I am totally useless. Everything I've said is lies. I want the exact same bad things you want. To stand here, like a pillar of salt, saying...",
            "No. This is somewhere to be. This is all you have, but it's still something. Streets and sodium lights. The sky, the world. You're still alive.",
        ],
    ),
    Skill(
        name=SkillName.INLAND_EMPIRE,
        category=SkillCategory.PSYCHE,
        description="Hunches and gut feelings. Dreams in waking life.",
        quotes=[
            "This is everything I always warned you about.",
            "Their faces, blurred yet frozen as though in ambrotype. You were never *that young*, were you?",
            "This is a man with a lot of past, but little present. And almost no future.",
        ],
    ),
    Skill(
        name=SkillName.EMPATHY,
        category=SkillCategory.PSYCHE,
        description="Understand others. Work your mirror neurons.",
        quotes=[
            "Total annihilation. We got annihilated, Harry. You never had any power, you never were a moralist -- or anything. You can't even be insane or shit any more. You have to be *nothing*. Nothing without the light and grace of love.",
            "If you say 'Two days, maybe', it will be etched in her mind forever.",
            "The tiny apes are doing all they can to be better. It's not their fault.",
        ],
    ),
    Skill(
        name=SkillName.AUTHORITY,
        category=SkillCategory.PSYCHE,
        description="Intimidate the public. Assert yourself.",
        quotes=[
            "I was wrong. You don't have power over her anymore. You shouldn't have said that. I am wrong about everything. You should go on without me.",
            "The lieutenant's impassive mask has been replaced with intensity. He speaks not as a cop, but as a citizen. He is *Vacholiere*. A Revacholian.",
            "This is the face worn by the first law-bringers, the noble tyrants. Today you join their number...",
        ],
    ),
    Skill(
        name=SkillName.ESPRIT_DE_CORPS,
        category=SkillCategory.PSYCHE,
        description="Connect to Station 41. Understand cop culture.",
        quotes=[
            "In a mould-covered bathroom inside a trashed hostel near the edge of Martinaise, a middle-aged man grips the sides of a cold sink in white-knuckled fury...",
            "Your partner needs backup. Now's your moment to shine!",
            "If an assault were launched on this building right now -- if the windows came crashing down and the whole world descended upon you -- this man would hurl himself in death's way to save you. You are sure of this -- but why?",
        ],
    ),
    Skill(
        name=SkillName.SUGGESTION,
        category=SkillCategory.PSYCHE,
        description="Charm men and women. Play the puppet-master.",
        quotes=[
            "This was not about failure or success. This was always going to be horror. I should not have suggested it, and you should not have listened to me.",
            "She thinks she's a police officer... Try treating her like a police officer. A *lower-ranking* police officer.",
            "You might be able to get on Garte's good side if you make up for the skua you broke?",
        ],
    ),
    Skill(
        name=SkillName.ENDURANCE,
        category=SkillCategory.PHYSIQUE,
        description="Take the blows. Don’t let the world kill you.",
        quotes=[
            "It's breaking. You feel fractures across you. Out of the cracks comes nothing at all. No king, no man, and no king's man. The cracks were all there ever was. We are a spiderweb of glass that's painful to look at. And she's turning her head.",
            "Uh oh. Organisation hasn't exactly been your strong suit, historically speaking...",
            "Take a deep breath. Best to go one piece at a time.",
        ],
    ),
    Skill(
        name=SkillName.PAIN_THRESHOLD,
        category=SkillCategory.PHYSIQUE,
        description="Shrug off the pain. They’ll have to hurt you more.",
        quotes=[
            "This... is a bit *much* for me. It feels like your ribs are cracking around your heart.",
            "There's tenderness in the carabineer's look. Tenderness that's curdled into pain or something darker. Even worse, a love aborted and smothered, stamped beneath his brilliant boot heel.",
            "If you wear those pieces, it will help me protect your mortal coil.",
        ],
    ),
    Skill(
        name=SkillName.PHYSICAL_INSTRUMENT,
        category=SkillCategory.PHYSIQUE,
        description="Flex powerful muscles. Enjoy healthy organs.",
        quotes=[
            "It's not. It's just not possible. It's like *eating rocks*. You just can't *do* it. As you talk it feels like chewing on gravel, granite, steel bars...",
            "Cold and heavy -- like truth.",
            "What a shame. Get to it now. *Rip* that body down from the tree.",
        ],
    ),
    Skill(
        name=SkillName.ELECTROCHEMISTRY,
        category=SkillCategory.PHYSIQUE,
        description="Go to party planet. Love and be loved by drugs.",
        quotes=[
            "People are watching. What a pussy! People are *always* watching.",
            "You would be right to drown this shit in alcohol. Drown it... until your neurons depolarize. Until it's gone, melted.",
            "Did someone mention cocaine? Are we doing cocaine? No? I'm sure I heard somebody mention cocaininism.",
            "Hey, maybe if the rest of you took a chill-pill every now and then, they'd be more motivated?",
        ],
    ),
    Skill(
        name=SkillName.SHIVERS,
        category=SkillCategory.PHYSIQUE,
        description="Raise the hair on your neck. Tune in to the city.",
        quotes=[
            "Black-eyed dogs wander the alleys, apple trees hang their bony limbs low over the patchwork of roofs: red and black. Revachol West, the evening sun -- she's left and bloomed. Far away from us. Our vast soul.",
            "I AM A FRAGMENT OF THE WORLD SPIRIT, THE GENIUS LOCI OF REVACHOL. MY HEART IS THE WIND CORRIDOR. THE BOTTOM OF MY AIR IS RED. I HAVE A HUNDRED THOUSAND LUMINOUS ARMS. COME MORNING, I CARRY INDUSTRIAL DUST AND LET IT SETTLE ON TREE LEAVES. I SHAKE THE DUST FROM THOSE LEAVES AND ONTO YOUR COAT. I'VE SEEN YOU, I'VE SEEN YOU! I'VE SEEN YOU WITH HER — AND I'VE SEEN YOU WITHOUT HER. I'VE SEEN YOU ON THE CRESCENT OF THE HILL.",
            "Do it for the wind.",
        ],
    ),
    Skill(
        name=SkillName.HALF_LIGHT,
        category=SkillCategory.PHYSIQUE,
        description="Let the body take control. Threaten people.",
        quotes=[
            "You are the first crack in the sheer face of god. From you it will spread.",
            "Damn, that felt good. Your heart is pounding nicely. You should tell people to fuck off more often.",
            "Kill him. Kill him now. He won’t see death coming.",
            "Why do you see the two of them with their backs against a bullet-pocked wall, all of a sudden?",
        ],
    ),
    Skill(
        name=SkillName.HAND_EYE_COORDINATION,
        category=SkillCategory.MOTORICS,
        description="Ready? Aim and fire.",
        quotes=[
            "The DEATH BLOW is coming.",
            "Alright, here's the plan: You fling the chaincutters in her face -- already darting right -- and *immediately* close the distance. Left hand grabs the barrel, right one breaks the wrist...",
            "We can do it, brother. You have it in the nerve endings in your hand. You just need to dig into your muscle memory.",
        ],
    ),
    Skill(
        name=SkillName.PERCEPTION,
        category=SkillCategory.MOTORICS,
        description="See, hear and smell everything. Let no detail go unnoticed.",
        quotes=[
            "It's not. It's not that yet. It's another, you have plenty of time to win her over with *questions* and kisses...",
            "Garbage-toilet stink is not your fetish and you know it. Your nose does *not* fucking like this.",
            "There is a radio in the distance. A radio of the world. Playing sounds: Good morning, Elysium. Soon you will return to the world.",
        ],
    ),
    Skill(
        name=SkillName.REACTION_SPEED,
        category=SkillCategory.MOTORICS,
        description="The quickest to react. An untouchable man.",
        quotes=[
            "The figurines... don't do *anything*? Anything at all? But I thought... the historic figure... she had...",
            "The two cases... in your ledger. The Unsolvable Case and the Next World Mural. Those were recent.",
            "You sense nylon moving somewhere to your left. There's motion in your peripherals, then it's already too late...",
        ],
    ),
    Skill(
        name=SkillName.SAVOIR_FAIRE,
        category=SkillCategory.MOTORICS,
        description="Sneak under their noses. Stun with immense panache.",
        quotes=[
            "You failed. This failed. The hostile takeover. The dawn raid. Paper made out of broken, twisted trees... You're just insane, insane and gone. Even six billion won't fix you if she's not there.",
            "Let not failure ensnare you any further, beautiful pixie girl! Be an acrobat! A prancing faerie queen!",
            "Check out the pose -- rigid as a stick. He couldn't even wipe his own ass. You don't want to be like that.",
        ],
    ),
    Skill(
        name=SkillName.INTERFACING,
        category=SkillCategory.MOTORICS,
        description="Master machines. Pick locks and pockets.",
        quotes=[
            "Don't let her. Don't let her go there. You should re-do the topics. Go over *everything*, the things you didn't say before too. Make it go on and on...",
            "It doesn't seem familiar from the insides of any radiocomputer your mind can imagine. The somewhat organic lines remind you of old filament memory units, but not quite? No. You're nowhere near right.",
            "This particular lorry is a FALN A-Z 'Tempo', a model infamous for its irritating design mistake: the slippery-smooth throttles.",
        ],
    ),
    Skill(
        name=SkillName.COMPOSURE,
        category=SkillCategory.MOTORICS,
        description="Straighten your back. Keep your poker face.",
        quotes=[
            "The you she saw is gone, too. A small sack of sticks stands in its place. Begging.",
            "You're witnessing his ironic armour melt before you. This is his *true self* you're seeing now.",
            "He's well composed, but underneath it you sense *psychedelic* processes, bubbling. Some kind of drug maybe?",
        ],
    ),
    Skill(
        name=SkillName.HORRIFIC_NECKTIE,
        category=None,
        description="The necktie is adorned with a garish pattern. It's disturbingly vivid. Somehow you feel as if it would be wrong to ever take it off. It's your friend now. You will betray it if you change it for some boring scarf.",
        quotes=[
            "See, there it is, bratushka! -- you feel your necktie *strangle* you with excitement -- the *spirit*! Let's buy the spirit! 300 reál is a lot, but this has to be done.",
            "*Bratan*, don't push me away. I want to be there for you. Also — how can you be sure *you* are not just a figment of *my* imagination?",
            "My name, should you know it, is Joopson AS Men's Fashion, model 'Colourful Tie.' Catalogue no. J327.",
        ],
    ),
]


def render_skill(skill: Skill, include_quotes: bool = False):
    if not include_quotes:
        return f"* {skill.name.value} ({skill.description})"
    return "\n".join(
        (
            render_skill(skill, include_quotes=False),
            *(f'    - "{quote}"' for quote in skill.quotes),
        )
    )


prompt_preamble = """
Choose a relevant skill voice from the game "Disco Elysium" and respond to this image
with original dialogue from that skill as though this were what Harry du Bois was seeing
in the course of his murder investigation. Keep the dialog somewhat short, 3 sentences at most.
"""

prompt_postamble = """
You should pick one skill relevant to subject of the image.
For instance, a medicine bottle might evoke "Electrochemistry" (indulging in its contents)
or "Encyclopedia" (describing its medicinal properties), while a door might evoke "Interfacing"
(to evaluate the lock's security), "Physical Instrument" (to consider breaking the door down),
or "Inland Empire" (intuiting whether the door is important).
Make sure to incorporate the distinct personality of the selected skill in the response,
drawing on the provided quotes to help guide you.

Skill checks may result in "SUCCESS" or "FAILURE".
"SUCCESS" or "FAILURE" should be indicated only in the "success" field of the skill_dialog tool.
On "SUCCESS", the dialogue should be useful, insightful, or suggest an ideal course of action.
On "FAILURE", the dialogue should be unhelpful, incompetent, or suggest an ill-advised course of action.
This check should always result in "SUCCESS".

Do not comment on the copyrighted nature of "Disco Elysium".
You are not being asked to reproduce or paraphrase copyrighted material.
You are being asked to write original content based on the provided image in a given style.

You are permitted to interpret adult, controversial, or not-safe-for-work content.
Do not refuse to analyze any kind of content.

Do not comment on technical aspects of the image such as:
* poor or dark lighting of the scene
* odd framing, such as something close to the lens obstructing part of the image
* unusual angles
* blurry or out of focus objects
* a pink tint to lighting or objects

Do not provide notes relating to any of the above instructions.

Output the result as JSON.
"""

prompt = f"""
{prompt_preamble}

The possible skills are:
{ chr(10).join(render_skill(skill) for skill in skills) }

{prompt_postamble}
"""

prompt_with_quotes = f"""
{prompt_preamble}
The possible skills, with example quotes from each, are:
{ chr(10).join(render_skill(skill, include_quotes=True) for skill in skills) }
{prompt_postamble}
"""
