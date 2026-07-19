from api.dependencies import create_container


def main():

    container = create_container()

    try:

        agent = container.agent_service

        decision = agent.decide(
            "Download the employee handbook."
        )

        print()
        print("=" * 80)
        print("LIVE AGENT TEST")
        print("=" * 80)
        print()

        print("Intent   :", decision.intent)
        print("Departments  :", decision.departments)
        print("Confidence   :", decision.confidence)
        print("Metadata     :", decision.metadata)

        print()
        print("=" * 80)
        print("FINAL STATUS: PASS")
        print("=" * 80)

    finally:
        container.close()


if __name__ == "__main__":
    main()