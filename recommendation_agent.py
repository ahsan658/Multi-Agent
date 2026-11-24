import os
from crewai import Agent, Task, Crew, LLM


def run_recommendation_agent(applicant_profile, programs):

    llm = LLM(
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    agent = Agent(
        role="Academic Program Recommendation Specialist",
        goal="Recommend suitable university programs to students.",
        backstory=(
            "You are an academic advisor familiar with "
            "Pakistani university programs and qualifications."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=f"""
        Recommend suitable university programs.

        STUDENT PROFILE:
        {applicant_profile}

        AVAILABLE PROGRAMS:
        {programs}

        Consider:
        1. Academic qualification
        2. Field or major
        3. CGPA
        4. Percentage
        5. Target degree
        6. Academic interests
        7. Career goals

        Explain:
        - Why the program matches the student.
        - How it relates to academic interests.
        - Which requirements need further verification.

        Rules:
        - Do not guarantee admission.
        - Never invent university requirements.
        - Use only the provided program data.
        """,
        expected_output=(
            "A list of recommended university programs "
            "with explanations."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    return str(crew.kickoff())
