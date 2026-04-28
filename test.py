from debatev1 import republican_node,DebateState,AIMessage

state = DebateState(
    question="Should taxes be increased on the wealthy?",
    messages=[
        AIMessage(content="We should raise taxes to reduce inequality.", name="democrat"),
        AIMessage(content="Higher taxes hurt economic growth.", name="republican"),
        AIMessage(content="Growth means nothing without fairness.", name="democrat"),
    ]
)

result = republican_node(state)
print(result)