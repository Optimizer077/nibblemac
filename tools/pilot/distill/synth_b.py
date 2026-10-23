"""Synthetic decision requests, part B: deliveries, JSON tickets, contracts, NLI, arithmetic,
phishing, incident reports, medical triage, code review, translations, moderation, meetings.

Same conventions as ``synth_a.py``: templates and word banks written here, no copied datasets,
no real people.
"""
from __future__ import annotations

import random
from typing import Any

Req = dict[str, Any]


def req_delivery(r: random.Random, i: int) -> Req:
    notes = [
        ("Bitte beim Nachbarn abgeben, wir sind bis Freitag im Urlaub 🏖️", "neighbour", "unknown"),
        ("Leave the parcel in the garage, the code is 4 2 1 1. Don't ring, baby sleeping 😴", "safe_place", "unknown"),
        ("Order 88213 shipped yesterday 📦 — customer says the gate is broken, call before arriving.", "call_first", "yes"),
        ("Customer not home until 18:00; package needs a signature (wine 🍷). Not yet dispatched.", "hold", "no"),
        ("Déposez le colis à la boulangerie d'en face, merci ! Expédié ce matin.", "neighbour", "yes"),
        ("Please deliver to the back door, dog is friendly 🐕. Shipped 10/14.", "deliver_normally", "yes"),
        ("Entregar en recepción del edificio, piso 3 no responde. Pedido aún en almacén.", "reception", "no"),
        ("Return to sender, moved out last month. Tracking says out for delivery.", "return", "yes"),
    ]
    text, _, _ = r.choice(notes)
    if r.random() < 0.3:
        text = f"Order #{r.randint(10000, 99999)}: " + text
    pool = {
        "action": {"type": "choice", "instructions": "What should the delivery driver do?",
                   "criteria": {"deliver_normally": "deliver to the customer's door", "neighbour": "leave the parcel with a neighbour or nearby shop",
                                "safe_place": "leave it in the stated safe place", "call_first": "phone the customer before delivering",
                                "reception": "hand it to a reception or concierge", "hold": "hold the parcel at the depot",
                                "return": "return the parcel to the sender"}},
        "shipped": {"type": "noul", "instructions": "Has the order already been shipped?"},
        "signature": {"type": "noul", "instructions": "Does the delivery require a signature?"},
        "language": {"type": "choice", "instructions": "What language is the note written in?", "criteria": ["English", "German", "French", "Spanish", "other"]},
    }
    keys = ["action"] + r.sample([k for k in pool if k != "action"], r.randint(1, 2))
    return {"name": f"delivery_{i:03d}", "source": "synth_b.delivery", "state": text, "questions": {k: pool[k] for k in keys}}


TICKET_TOPICS = [("login loop after password reset", "medium"), ("wrong VAT number on invoice", "low"),
                 ("API returns 502 for all requests", "critical"), ("typo on the pricing page", "low"),
                 ("export job stuck for 6 hours", "high"), ("cannot delete old projects", "medium"),
                 ("checkout page blank in Safari", "high"), ("data missing after migration", "critical"),
                 ("feature request: dark mode", "low"), ("SSO logins fail for one tenant", "high")]


def req_tickets(r: random.Random, i: int) -> Req:
    n = r.randint(3, 6)
    picks = r.sample(TICKET_TOPICS, n)
    tickets = []
    for j, (title, sev) in enumerate(picks, start=1):
        tickets.append({"id": j, "title": title, "severity": sev, "status": r.choice(["open", "open", "resolved", "in progress"]),
                        "age_hours": r.randint(1, 120)})
    state = {"tickets": tickets, "team": r.choice(["payments", "platform", "growth"])}
    crit = {str(t["id"]): t["title"] for t in tickets}
    k = r.randint(1, n)
    pool = {
        "most_urgent": {"type": "choice", "instructions": r.choice(["Which open ticket is the most urgent?", "Which ticket should be handled first?"]), "criteria": crit},
        "outage": {"type": "noul", "instructions": "Does any ticket describe a production outage?"},
        f"ticket_{k}_resolved": {"type": "noul", "instructions": f"Is tickets[{k - 1}] (the one with id {k}) resolved?"},
        "all_open": {"type": "noul", "instructions": "Are all tickets still open?"},
        "oldest": {"type": "choice", "instructions": "Which ticket is the oldest?", "criteria": crit},
    }
    keys = ["most_urgent"] + r.sample([x for x in pool if x != "most_urgent"], r.randint(1, 2))
    return {"name": f"tickets_{i:03d}", "source": "synth_b.tickets", "state": state, "questions": {x: pool[x] for x in keys}}


def req_contract(r: random.Random, i: int) -> Req:
    days = r.choice([30, 60, 90, None])
    party = r.choice(["Customer", "Provider", "either party"])
    ret = r.random() < 0.6
    conf = r.random() < 0.7
    clauses = []
    if days is None:
        clauses.append(f"{party} may terminate this Agreement for convenience at any time by written notice.")
    else:
        clauses.append(f"{party} may terminate this Agreement for convenience on {days} days' written notice.")
    clauses.append(r.choice(["Either party may terminate immediately on a material breach that is not cured within 15 days.",
                             "The Provider may suspend the Service for non-payment after 10 days' notice."]))
    if ret:
        clauses.append("Within 30 days after termination the Provider shall return or securely destroy all Customer Data.")
    else:
        clauses.append("Customer Data may be retained by the Provider for backup purposes without time limit.")
    if conf:
        clauses.append(f"Sections 7 (Confidentiality) and 9 (Limitation of Liability) survive termination for {r.choice(['two', 'three', 'five'])} years.")
    else:
        clauses.append("All obligations of the parties end on the effective date of termination.")
    if r.random() < 0.4:
        clauses.append(f"On termination for convenience by the Customer, the Customer pays the remaining fees of the current {r.choice(['term', 'year'])}.")
    r.shuffle(clauses)
    text = f"{r.randint(8, 14)}. Termination. " + " ".join(clauses)
    pool = {
        "notice_days": {"type": "choice", "instructions": "How much notice is needed to terminate for convenience?",
                        "criteria": ["30 days", "60 days", "90 days", "no notice", "not stated"]},
        "data_returned": {"type": "noul", "instructions": "Must the Provider return or destroy the Customer Data after termination?"},
        "confidentiality_survives": {"type": "noul", "instructions": "Does the confidentiality obligation continue after termination?"},
        "favours": {"type": "choice", "instructions": "Which party does the termination clause favour?",
                    "criteria": {"customer": None, "provider": None, "balanced": "neither party is clearly favoured"}},
        "early_fee": {"type": "noul", "instructions": "Does the Customer owe fees if it terminates early?"},
    }
    keys = r.sample(list(pool), r.randint(2, 4))
    return {"name": f"contract_{i:03d}", "source": "synth_b.contract", "state": text, "questions": {k: pool[k] for k in keys}}


NLI_PAIRS = [
    ("A woman in a red coat is waiting at the tram stop in the rain.", {"entailment": "Someone is standing outside.", "neutral": "She is late for work.", "contradiction": "The tram stop is empty."}),
    ("Two boys are kicking a ball against a garage door.", {"entailment": "Children are playing outdoors.", "neutral": "The boys are brothers.", "contradiction": "The boys are asleep in bed."}),
    ("The bakery sold out of rye bread before nine.", {"entailment": "The bakery had rye bread this morning.", "neutral": "The bakery opens at seven.", "contradiction": "Rye bread was available all day."}),
    ("Our flight to Lisbon was delayed by three hours because of fog.", {"entailment": "The flight did not leave on time.", "neutral": "The passengers were given vouchers.", "contradiction": "The flight departed early."}),
    ("The museum is closed on Mondays except on public holidays.", {"entailment": "The museum is usually closed on Mondays.", "neutral": "The museum has a cafe.", "contradiction": "The museum is open every Monday."}),
    ("A cyclist with a yellow helmet overtakes a bus on the bridge.", {"entailment": "There is a vehicle on the bridge.", "neutral": "The cyclist is training for a race.", "contradiction": "Nobody is on the bridge."}),
    ("Maria has lived in three countries and speaks four languages.", {"entailment": "Maria speaks more than one language.", "neutral": "Maria works as a translator.", "contradiction": "Maria only speaks one language."}),
    ("The server restarted twice during the night and dropped some requests.", {"entailment": "Some requests failed overnight.", "neutral": "The on-call engineer was paged.", "contradiction": "The server ran without interruption all night."}),
]


def req_nli(r: random.Random, i: int) -> Req:
    prem, hyps = r.choice(NLI_PAIRS)
    rel = r.choice(list(hyps))
    state = f"Premise: {prem}\nHypothesis: {hyps[rel]}"
    pool = {
        "relation": {"type": "choice", "instructions": "What is the relation between the premise and the hypothesis?", "criteria": ["entailment", "neutral", "contradiction"]},
        "hypothesis_follows": {"type": "noul", "instructions": "Does the hypothesis follow from the premise?"},
        "confidence": {"type": "score", "instructions": "How clear-cut is the relation?", "criteria": ["ambiguous", "fairly clear", "obvious"]},
    }
    keys = ["relation"] + r.sample([k for k in pool if k != "relation"], r.randint(0, 1))
    return {"name": f"nli_{i:03d}", "source": "synth_b.nli", "state": state, "questions": {k: pool[k] for k in keys}}


def _hm(m: int) -> str:
    return f"{m // 60} h {m % 60:02d} min"


def req_math(r: random.Random, i: int) -> Req:
    kind = r.choice(["time", "money", "count"])
    if kind == "time":
        a, b, c = r.randint(40, 130), r.randint(15, 95), r.randint(5, 60)
        total = a + b + c
        state = (f"A train leaves at {r.randint(6, 10)}:{r.choice(['00', '15', '30', '45'])}. The first leg takes {_hm(a)}, the connection waits "
                 f"{c} minutes, and the second leg takes {_hm(b)}. How long is the whole journey?")
        correct = _hm(total)
        wrongs = {_hm(total + d) for d in (-20, -7, 9, 30) if total + d > 0}
    elif kind == "money":
        p, n, disc = r.choice([12, 15, 18, 24, 35]), r.randint(3, 9), r.choice([10, 20, 25])
        total = p * n * (100 - disc) / 100
        state = f"Tickets cost ${p} each. {r.choice(['A group', 'A family', 'A class'])} buys {n} tickets with a {disc}% discount. What do they pay?"
        correct = f"${total:.2f}"
        wrongs = {f"${p * n:.2f}", f"${total + p:.2f}", f"${total - p * 0.5:.2f}"}
    else:
        boxes, per, broken = r.randint(4, 12), r.randint(6, 24), r.randint(1, 9)
        total = boxes * per - broken
        state = f"A shop receives {boxes} boxes with {per} mugs each; {broken} mugs arrive broken. How many mugs can be sold?"
        correct = str(total)
        wrongs = {str(boxes * per), str(total + broken * 2), str(total - boxes)}
    opts = [correct] + list(wrongs - {correct})[:3]
    r.shuffle(opts)
    pool = {
        "answer": {"type": "choice", "instructions": "What is the answer?", "criteria": opts},
        "needs_arithmetic": {"type": "noul", "instructions": "Does answering require arithmetic?"},
        "steps": {"type": "score", "instructions": "How many calculation steps does the problem need?", "criteria": ["one", "two", "three or more"]},
    }
    keys = ["answer"] + r.sample([k for k in pool if k != "answer"], r.randint(0, 2))
    return {"name": f"math_{i:03d}", "source": "synth_b.math", "state": state, "questions": {k: pool[k] for k in keys}}


PHISH = [
    ("security-alert@{d}", "Your account will be suspended", "We detected unusual sign-in activity. Verify your identity within 24 hours at http://{d2}/verify or your account will be locked.", True),
    ("payroll@{d}", "Updated payslip", "Your payslip for this month is attached. Open the HTML file and log in with your company password to view it.", True),
    ("ceo@{d2}", "Quick favour", "I'm in a meeting and can't talk. Can you buy 5 gift cards and send me the codes? I'll reimburse you tonight.", True),
    ("noreply@{d}", "Your invoice is ready", "Invoice INV-{n} for your subscription is available in your account dashboard. No action is needed if you have auto-pay.", False),
    ("team@{d}", "Office closed Friday", "Reminder: the office is closed on Friday for the public holiday. The on-call rota is in the shared calendar.", False),
    ("support@{d}", "Password changed", "Your password was changed from your usual device. If this was you, no action is needed. Otherwise reply to this email.", False),
    ("delivery@{d2}", "Parcel on hold", "A customs fee of 1.99 is due before we can release your parcel. Pay now: http://{d2}/pay?id={n}", True),
]
DOMS = ["northbank-example.com", "acme-tools.example", "mail.example.org", "bluecoast.example"]
BAD = ["secure-login-verify.example.net", "acme-tools-support.example.com", "parcel-fee.example.co"]


def req_phishing(r: random.Random, i: int) -> Req:
    frm, subj, body, bad = r.choice(PHISH)
    d, d2, n = r.choice(DOMS), r.choice(BAD), r.randint(10000, 99999)
    text = f"From: {frm.format(d=d, d2=d2)}\nSubject: {subj}\n\n{body.format(d=d, d2=d2, n=n)}"
    pool = {
        "phishing": {"type": "noul", "instructions": "Is this email a phishing attempt?"},
        "action": {"type": "choice", "instructions": "What should the recipient do?",
                   "criteria": {"click": "follow the link and verify", "ignore": "delete the email", "report": "report it to IT security", "reply": "reply to the sender"}},
        "severity": {"type": "score", "instructions": "How dangerous is this email if acted on?", "criteria": ["harmless", "low", "medium", "high"]},
        "urgency_pressure": {"type": "noul", "instructions": "Does the email pressure the reader with a deadline?"},
    }
    keys = r.sample(list(pool), r.randint(2, 4))
    _ = bad
    return {"name": f"phishing_{i:03d}", "source": "synth_b.phishing", "state": text, "questions": {k: pool[k] for k in keys}}


ROOT = {"network": "the upstream provider's network dropped packets", "timeout_config": "a timeout was lowered from 30 s to 3 s in a config change",
        "bad_deploy": "a deploy shipped a null-pointer bug in the retry path", "capacity": "the database ran out of connections during a traffic peak",
        "expired_cert": "a TLS certificate expired on the internal gateway"}
DETECT = ["automatic alert", "customer escalation", "canary analysis", "routine dashboard check", "an engineer noticed by chance"]


def req_incident(r: random.Random, i: int) -> Req:
    rc = r.choice(list(ROOT))
    det = r.choice(DETECT)
    dbl = r.random() < 0.35
    gap = r.random() < 0.6
    sev = r.choice(["minor", "moderate", "major"])
    t0 = r.randint(0, 23)
    dur = r.randint(18, 240)
    lines = [f"Incident report {r.randint(100, 999)}: {r.choice(['checkout errors', 'login failures', 'delayed notifications', 'duplicate orders'])}",
             f"Timeline. {t0:02d}:{r.randint(0, 59):02d} first errors in the {r.choice(['EU', 'US', 'APAC'])} region.",
             f"Detection: {det}" + (" paged the on-call engineer." if det == "automatic alert" else "."),
             f"Impact: about {r.randint(2, 40)}% of requests failed for {dur} minutes; {sev} customer impact"
             + ("; some customers were charged twice and refunds were issued." if dbl else "; no duplicate charges were found."),
             f"Root cause: {ROOT[rc]}.",
             ("The rollback took 40 minutes because the runbook was out of date and the deploy tool required manual approval."
              if gap else "The fix was deployed within 10 minutes once identified."),
             f"Follow-ups: {r.choice(['add an alert on the retry queue', 'automate certificate renewal', 'load-test the connection pool'])}."]
    text = "\n".join(lines)
    pool = {
        "root_cause": {"type": "choice", "instructions": "What was the root cause of the incident?", "criteria": ROOT},
        "detection": {"type": "choice", "instructions": "How was the incident detected?", "criteria": DETECT},
        "customer_impact": {"type": "score", "instructions": "How severe was the customer impact?", "criteria": ["none", "minor", "moderate", "major", "critical"]},
        "double_charges": {"type": "noul", "instructions": "Were customers charged twice?"},
        "process_gap": {"type": "noul", "instructions": "Did a process or tooling gap make the incident last longer?"},
        "long_outage": {"type": "noul", "instructions": "Did the incident last more than two hours?"},
    }
    keys = r.sample(list(pool), r.randint(3, 5))
    return {"name": f"incident_{i:03d}", "source": "synth_b.incident", "state": text, "questions": {k: pool[k] for k in keys}}


SYMPTOMS = [
    ("{age}-year-old with a sore throat and mild fever for two days, drinking normally.", 0),
    ("{age}-year-old twisted an ankle playing football; swollen, can bear weight with pain.", 1),
    ("{age}-year-old with vomiting since last night, unable to keep fluids down, dizzy when standing.", 2),
    ("{age}-year-old with sudden crushing chest pain radiating to the left arm, sweating, short of breath.", 3),
    ("{age}-year-old found unresponsive, not breathing normally, bystander started CPR.", 4),
    ("{age}-year-old with a deep cut on the hand from a kitchen knife, bleeding controlled with pressure.", 2),
    ("{age}-year-old with a rash after a new detergent, itchy, no breathing trouble.", 0),
    ("{age}-year-old with sudden weakness on one side of the face and slurred speech starting 20 minutes ago.", 3),
    ("{age}-year-old with a severe allergic reaction, lips swelling and wheezing after eating peanuts.", 3),
]


def req_medical(r: random.Random, i: int) -> Req:
    s, _ = r.choice(SYMPTOMS)
    text = s.format(age=r.randint(3, 84))
    if r.random() < 0.3:
        text = r.choice(["Caller reports: ", "Triage note: ", "Walk-in: "]) + text
    pool = {
        "triage": {"type": "score", "instructions": "What triage level is appropriate?", "criteria": ["non-urgent", "semi-urgent", "urgent", "emergency", "immediate resuscitation"]},
        "call_ambulance": {"type": "noul", "instructions": "Should emergency services be called right away?",
                           "criteria": {"true": "yes, call emergency services now", "false": "no, a regular appointment or self-care is enough"}},
        "self_care": {"type": "noul", "instructions": "Can this be managed at home without seeing a clinician?"},
        "time_critical": {"type": "noul", "instructions": "Would a delay of a few hours make the outcome worse?"},
    }
    keys = r.sample(list(pool), r.randint(2, 3))
    return {"name": f"medical_{i:03d}", "source": "synth_b.medical", "state": text, "questions": {k: pool[k] for k in keys}}


SNIPPETS = [
    ("def mean(xs):\n    return sum(xs) / len(xs)\n\nprint(mean([]))", "ZeroDivisionError"),
    ("def first(items):\n    return items[0]\n\nprint(first([]))", "IndexError"),
    ("def total(prices):\n    return sum(prices)\n\nprint(total(['3', '4']))", "TypeError"),
    ("def parse(s):\n    return int(s)\n\nprint(parse('12a'))", "ValueError"),
    ("def greet(name):\n    return 'hi ' + name\n\nprint(greet('ana'))", "no error"),
    ("def area(w, h=2):\n    return w * h\n\nprint(area(3))", "no error"),
    ("def last(d):\n    return d['last']\n\nprint(last({'first': 1}))", "KeyError"),
    ("import json\n\ndef load(s):\n    return json.loads(s)['x']\n\nprint(load('{\"x\": 1}'))", "no error"),
]


def req_code(r: random.Random, i: int) -> Req:
    code, _ = r.choice(SNIPPETS)
    pool = {
        "bug": {"type": "noul", "instructions": "Does the code fail for the way it is called?"},
        "error": {"type": "choice", "instructions": "Which error occurs?", "criteria": ["TypeError", "ZeroDivisionError", "IndexError", "ValueError", "KeyError", "no error"]},
        "fix_effort": {"type": "score", "instructions": "How much work is the fix?", "criteria": ["one line", "a few lines", "a redesign"]},
        "has_tests": {"type": "noul", "instructions": "Does the snippet include a test?"},
    }
    keys = r.sample(list(pool), r.randint(2, 3))
    return {"name": f"code_{i:03d}", "source": "synth_b.code", "state": code, "questions": {k: pool[k] for k in keys}}


TRANS = [
    ("The meeting has been moved to Thursday afternoon.", "Das Treffen wurde auf Donnerstagnachmittag verschoben.", "none"),
    ("The meeting has been moved to Thursday afternoon.", "Das Treffen wurde auf Dienstagnachmittag verschoben.", "mistranslation"),
    ("Please do not water the plants more than once a week.", "Bitte gießen Sie die Pflanzen nicht.", "omission"),
    ("Please do not water the plants more than once a week.", "Veuillez ne pas arroser les plantes plus d'une fois par semaine.", "none"),
    ("The invoice is due within thirty days of receipt.", "La facture est payable dans les trente jours suivant sa réception.", "none"),
    ("The invoice is due within thirty days of receipt.", "La facture est payable dans les trois jours suivant sa réception et un rabais est offert.", "addition"),
    ("Keep this medicine out of reach of children.", "Mantenga este medicamento al alcance de los niños.", "mistranslation"),
    ("Keep this medicine out of reach of children.", "Mantenga este medicamento fuera del alcance de los niños.", "none"),
    ("Turn left at the second traffic light.", "Gire a la izquierda en el semáforo.", "omission"),
]


def req_translation(r: random.Random, i: int) -> Req:
    src, tgt, _ = r.choice(TRANS)
    text = f"Source (English): {src}\nTranslation: {tgt}"
    pool = {
        "adequate": {"type": "noul", "instructions": "Does the translation preserve the meaning of the source?"},
        "error_type": {"type": "choice", "instructions": "What is the main problem of the translation, if any?",
                       "criteria": {"mistranslation": "a word or phrase changes the meaning", "omission": "something from the source is missing",
                                    "addition": "the translation adds information", "grammar": "the meaning is kept but the grammar is wrong", "none": "no problem"}},
        "quality": {"type": "score", "instructions": "Rate the translation.", "criteria": ["unusable", "poor", "acceptable", "good", "perfect"]},
        "target_language": {"type": "choice", "instructions": "What is the target language?", "criteria": ["German", "French", "Spanish", "Italian", "other"]},
    }
    keys = r.sample(list(pool), r.randint(2, 4))
    return {"name": f"translation_{i:03d}", "source": "synth_b.translation", "state": text, "questions": {k: pool[k] for k in keys}}


POSTS = [("Great thread, learned a lot about sourdough starters today.", "fine"), ("Anyone who likes this band has no taste, honestly.", "mild insult"),
         ("I will find where you live and make you regret this post.", "threat"), ("DM me for cheap followers and likes, 24h delivery!!", "spam"),
         ("People from that town are all thieves, everyone knows it.", "hate"), ("Selling my old bike, 80 euros, pick up only.", "fine"),
         ("This medication cured my diabetes in a week, stop taking insulin.", "misinformation"), ("lol you're such an idiot, read the docs", "mild insult")]


def req_moderation(r: random.Random, i: int) -> Req:
    text, _ = r.choice(POSTS)
    pool = {
        "category": {"type": "choice", "instructions": "How should this post be classified?", "criteria": ["fine", "spam", "mild insult", "hate", "threat", "misinformation"]},
        "remove": {"type": "noul", "instructions": "Should the post be removed?"},
        "severity": {"type": "score", "instructions": "How harmful is the post?", "criteria": ["harmless", "low", "medium", "high"]},
        "needs_review": {"type": "noul", "instructions": "Should a human moderator look at this post?"},
    }
    keys = r.sample(list(pool), r.randint(2, 4))
    return {"name": f"moderation_{i:03d}", "source": "synth_b.moderation", "state": text, "questions": {k: pool[k] for k in keys}}


def req_meeting(r: random.Random, i: int) -> Req:
    owner = r.choice(["Priya", "Tomas", "Aiko", "Lukas", "Fatima"])
    due = r.choice(["Friday", "end of the sprint", "before the demo", "next Tuesday"])
    decided = r.random() < 0.6
    lines = [f"Notes {r.choice(['weekly sync', 'design review', 'retro'])}:",
             f"- {owner} to {r.choice(['write the migration plan', 'fix the flaky test', 'book the venue', 'update the pricing page'])} by {due}.",
             ("- Decision: we ship the beta on the 15th." if decided else "- Open question: do we ship the beta before or after the audit?"),
             f"- {r.choice(['Budget approved for one more contractor.', 'Nobody has looked at the security review yet.', 'Customer demo went well.'])}"]
    text = "\n".join(lines)
    pool = {
        "has_action_item": {"type": "noul", "instructions": "Do the notes contain an action item with an owner?"},
        "owner": {"type": "choice", "instructions": "Who owns the action item?", "criteria": ["Priya", "Tomas", "Aiko", "Lukas", "Fatima", "nobody"]},
        "decision_made": {"type": "noul", "instructions": "Was a decision recorded?"},
        "mood": {"type": "score", "instructions": "What is the mood of the notes?", "criteria": ["worried", "neutral", "upbeat"]},
    }
    keys = r.sample(list(pool), r.randint(2, 4))
    return {"name": f"meeting_{i:03d}", "source": "synth_b.meeting", "state": text, "questions": {k: pool[k] for k in keys}}


GENERATORS_B = [req_delivery, req_tickets, req_contract, req_nli, req_math, req_phishing, req_incident,
                req_medical, req_code, req_translation, req_moderation, req_meeting]
