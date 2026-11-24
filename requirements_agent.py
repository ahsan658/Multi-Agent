import os
from crewai import Agent, Task, Crew, LLM


def run_requirements_agent(applicant_profile, programs):

    llm = LLM(
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    agent = Agent(
        role="University Admission Requirements Analyst",
        goal="Analyze university admission requirements.",
        backstory=(
            "You are an admission requirements specialist "
            "familiar with the Pakistani education system."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=f"""
        Analyze the university admission requirements.

        STUDENT PROFILE:
        {applicant_profile}

        AVAILABLE PROGRAMS:
        {programs}

        Explain:
        1. Required qualification
        2. Minimum CGPA or percentage
        3. Required academic background
        4. Entry test requirements

        Rules:
        - Use only the provided program data.
        - Never invent admission requirements.
        - Mention missing information clearly.
        - Consider the Pakistani education system.
        """,
        expected_output=(
            "A program-by-program admission requirements report."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    return str(crew.kickoff())
