import os
from crewai import Agent, Task, Crew, LLM


def run_eligibility_agent(applicant_profile, programs):

    llm = LLM(
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    agent = Agent(
        role="University Eligibility Evaluator",
        goal="Evaluate student eligibility for university programs.",
        backstory=(
            "You are an academic eligibility evaluator "
            "familiar with Pakistani university admission criteria."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=f"""
        Evaluate the student's eligibility.

        STUDENT PROFILE:
        {applicant_profile}

        AVAILABLE PROGRAMS:
        {programs}

        For each relevant program, classify the student as:

        1. Appears Eligible
        2. Appears Not Eligible
        3. Cannot Determine

        Consider:
        - Academic qualification
        - Field or major
        - CGPA
        - Percentage
        - Target degree
        - Entry test

        Rules:
        - Use only provided requirements.
        - Do not invent admission criteria.
        - Missing information must not automatically
          be considered eligible.
        - Explain the reason for each decision.
        """,
        expected_output=(
            "A program-by-program eligibility assessment "
            "with status and explanation."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    return str(crew.kickoff())
