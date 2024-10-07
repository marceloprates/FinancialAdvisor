import os
import plotly
import pandas as pd
import streamlit as st
from skrub import deduplicate
from sqlalchemy import create_engine
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

api_key = os.getenv("api_key")


def infer_asset_class(row):
    """
    Infers the asset class based on the 'Sector' and 'Name' fields of the given row.
    Parameters:
    row (dict): A dictionary representing a row of data with at least 'Sector' and 'Name' keys.
    Returns:
    str: The inferred asset class, which can be 'ETFs', 'Bonds', 'Stocks', or 'Cash'.
    """

    row["Sector"] = str(row["Sector"])
    row["Name"] = str(row["Name"])
    if "ETF" in row["Sector"] or "ETF" in row["Name"]:
        return "ETFs"
    elif "Bond" in row["Name"]:
        return "Bonds"
    elif "Stock" in row["Sector"] or "Inc." in row["Name"]:
        return "Stocks"
    else:
        return "Cash"


def create_database():
    """
    Sanitize & process financial data for clients and advisors, and save to CSV and SQLite files.

    This function performs the following steps:
    1. Reads CSV files into pandas DataFrames.
    2. Deduplicates and validates client data.
    3. Infers asset classes for current allocations.
    4. Ensures each client has a row for each asset class.
    5. Computes current allocation percentages.
    6. Saves the processed DataFrames to CSV and SQLite files.

    The function reads data from the following CSV files:
    - data/client_target_allocations.csv
    - data/financial_advisor_clients.csv

    The processed data is saved to:
    - data/client_target_allocations.csv
    - data/financial_advisor_clients.csv
    - data/current_allocation.csv
    - SQLite database: data/financial_database.sqlite

    Returns:
        None
    """

    # Step 1: Read the CSV files into pandas DataFrames
    client_target_allocations = (
        pd.read_csv("data/client_target_allocations.csv")
        .reset_index(drop=True)
        .rename(columns={"Target Allocation (%)": "Target_Allocation"})
    )
    financial_advisor_clients = pd.read_csv(
        "data/financial_advisor_clients.csv"
    ).reset_index(drop=True)

    # Step 2:
    # - Use skrub's "deduplicate" to fix typos and inconsistencies in the data
    financial_advisor_clients["Client"] = deduplicate(
        financial_advisor_clients["Client"]
    ).tolist()
    # - Validate that the 'Client' column in the 'financial_advisor_clients' DataFrame matches the 'Client' column in the 'client_target_allocations' DataFrame
    financial_advisor_clients = financial_advisor_clients[
        financial_advisor_clients["Client"].isin(client_target_allocations["Client"])
    ]
    # Replace spaces with underscores in column names for all DataFrames
    for df in [
        client_target_allocations,
        financial_advisor_clients,
    ]:
        df.columns = df.columns.str.replace(" ", "_")

    # Step 3: Create 'current_allocation' DataFrame and infer the asset class for each row
    current_allocation = financial_advisor_clients.copy().reset_index(drop=True)
    current_allocation["Asset_Class"] = current_allocation.apply(
        infer_asset_class, axis=1
    )

    # Step 4: Iterate through clients and asset classes to ensure each client has a row for each asset class
    asset_classes = current_allocation["Asset_Class"].unique()
    clients = current_allocation["Client"].unique()
    for client in clients:
        for asset_class in ["ETFs", "Bonds", "Stocks", "Cash"]:
            if not (
                (current_allocation["Client"] == client)
                & (current_allocation["Asset_Class"] == asset_class)
            ).any():
                # Create a new row with zeros for Quantity and Current_Price
                new_row = pd.DataFrame(
                    [
                        {
                            "Client": client,
                            "Asset_Class": asset_class,
                            "Quantity": 0,
                            "Current_Price": 0,
                        }
                    ]
                )
                # Concatenate the new row to the DataFrame
                current_allocation = pd.concat([current_allocation, new_row])

    # Step 5: Compute current allocation
    current_allocation = (
        current_allocation.groupby(["Client", "Asset_Class"])[["Market_Value"]]
        .sum()
        .reset_index()
    )
    current_allocation["Total_Market_Value"] = current_allocation.groupby("Client")[
        "Market_Value"
    ].transform("sum")
    current_allocation["Current_Allocation"] = (
        current_allocation["Market_Value"] / current_allocation["Total_Market_Value"]
    ) * 100

    # Step 6: Save the DataFrames to CSV and sqlite files
    client_target_allocations.to_csv("data/client_target_allocations_2.csv")
    financial_advisor_clients.to_csv("data/financial_advisor_clients_2.csv")
    current_allocation.to_csv("data/current_allocation_2.csv")

    engine = create_engine("sqlite:///data/financial_database.sqlite")

    # Step 3: Write the DataFrames to SQL tables
    client_target_allocations.to_sql(
        "client_target_allocations", con=engine, if_exists="replace", index=False
    )
    financial_advisor_clients.to_sql(
        "financial_advisor_clients", con=engine, if_exists="replace", index=False
    )
    current_allocation.to_sql(
        "current_allocation", con=engine, if_exists="replace", index=False
    )


def get_response(user_input, chat_history):
    """
    Processes the user input through the financial assistant and generates a response.
    Args:
        user_input (str): The input provided by the user.
        chat_history (list): The history of the chat conversation.
    Returns:
        dict: A dictionary containing the following keys:
            - 'answer' (str): The generated summary response.
            - 'df' (DataFrame): The DataFrame resulting from the query.
            - 'fig' (Figure): The figure generated from the query.
    """

    text2sqlprompt = f"""
    Given the following user query and chat history, generate a SQL query:
    
    User query:
    {user_input}
    """

    try:
        query, df, fig = st.session_state.financial_assistant.ask(
            text2sqlprompt, print_results=False, allow_llm_to_see_data=True
        )
    except Exception as e:
        query, df, fig = None, None, None

    if df is None:
        summary = "I'm sorry, I couldn't find any relevant information."
    else:
        summary = st.session_state.financial_assistant.generate_summary(
            user_input,
            df.iloc[:100],
        )

    response = {"answer": summary, "df": df, "fig": fig}

    return response


def show_chat_history():
    """
    Displays the chat history stored in the session state.
    Iterates through the chat history and displays each message based on its type.
    - If the message is an instance of AIMessage, it is displayed as an AI message.
    - If the message is an instance of HumanMessage, it is displayed as a Human message.
    - If the message contains a DataFrame, it is displayed as a table.
    - If the message contains a Plotly figure, it is displayed as a chart.
    Note:
        This function assumes that `st.session_state.chat_history` is a list of messages,
        where each message can be an instance of AIMessage, HumanMessage, or a dictionary
        containing a DataFrame and/or a Plotly figure.
    """

    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        else:
            if message["df"] is not None:
                st.dataframe(message["df"])
            if message["fig"] is not None:
                st.plotly_chart(message["fig"], key=i)


def configure_financial_assistant():
    """
    Configures the financial assistant by initializing and setting up the necessary components.
    This function performs the following steps:
    1. Defines the `FinancialAssistant` class, which inherits from `ChromaDB_VectorStore` and `OpenAI_Chat`.
    2. Initializes an instance of `FinancialAssistant` with the specified configuration.
    3. Connects the financial assistant to a SQLite database.
    4. Retrieves and trains the financial assistant on the schema of specified tables.
    5. Adds Data Definition Language (DDL) statements to create necessary tables in the database.
    6. Trains the financial assistant with specific SQL queries and documentation for handling user questions.
    The tables involved include:
    - `client_target_allocations`
    - `financial_advisor_clients`
    - `current_allocation`
    The function also trains the assistant to handle specific questions and generate appropriate SQL queries.
    Note:
    - The `api_key` and `model` parameters should be defined in the configuration.
    - The SQLite database file should be located at "data/financial_database.sqlite".
    Raises:
        Any exceptions raised during the initialization, connection, or training process.
    """

    class FinancialAssistant(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

    st.session_state.financial_assistant = FinancialAssistant(
        config={"api_key": api_key, "model": "gpt-4"}
    )

    st.session_state.financial_assistant.connect_to_sqlite(
        "data/financial_database.sqlite"
    )
    for table in [
        "client_target_allocations",
        "financial_advisor_clients",
        "current_allocation",
    ]:
        schema = st.session_state.financial_assistant.run_sql(
            f"PRAGMA table_info({table});"
        )
        st.session_state.financial_assistant.train(
            documentation=f"""
            For the {table} table, the columns are {', '.join([x[1] for x in schema])}
        """
        )

    st.session_state.financial_assistant.add_ddl(
        """
    CREATE TABLE client_target_allocations (
        Client TEXT,
        Target_Portfolio TEXT,
        Asset_Class TEXT,
        Target_Allocation INTEGER
    );
    """
    )

    st.session_state.financial_assistant.add_ddl(
        """
    CREATE TABLE financial_advisor_clients (
        Client TEXT,
        Symbol TEXT,
        Name TEXT,
        Sector TEXT,
        Quantity INTEGER,
        Buy_Price REAL,
        Current_Price REAL,
        Market_Value REAL,
        Purchase_Date TEXT,
        Dividend_Yield REAL,
        PE_Ratio REAL,
        Week_52_High REAL,
        Week_52_Low REAL,
        Analyst_Rating TEXT,
        Target_Price REAL,
        Risk_Level TEXT
    );
    """
    )

    st.session_state.financial_assistant.add_ddl(
        """
    CREATE TABLE current_allocation (
        id INTEGER PRIMARY KEY,
        Client TEXT,
        Asset_Class TEXT,
        Market_Value REAL,
        Total_Market_Value REAL,
        Current_Allocation REAL
    );
    """
    )

    st.session_state.financial_assistant.train(
        question="List unique clients",
        sql="SELECT DISTINCT Client FROM financial_advisor_clients;",
    )
    st.session_state.financial_assistant.train(
        documentation="If the user asks for clients, consider the Client column"
    )
    st.session_state.financial_assistant.train(
        documentation="There is no Target_Allocation_Percent column. Just Target_Allocation"
    )
    st.session_state.financial_assistant.train(
        documentation="Create bar plots when the user asks for comparison of current allocation to target allocation"
    )
    st.session_state.financial_assistant.train(
        question="Compare the current allocation on ETF to the target allocation for Client1",
        sql="""
    SELECT 
        ca.Client,
        ca.Asset_Class,
        ca.Current_Allocation,
        cta.Target_Allocation,
        (ca.Current_Allocation - cta.Target_Allocation) AS Allocation_Difference
    FROM 
        current_allocation ca
    JOIN 
        client_target_allocations cta
    ON 
        ca.Client = cta.Client AND ca.Asset_Class = cta.Asset_Class
    WHERE 
        ca.Client = 'Client_1' AND ca.Asset_Class = 'ETFs';
    """,
    )


def main():
    """
    Main function to run the AI Financial Assistant application.
    This function sets up the Streamlit interface, initializes the financial assistant,
    and handles user interactions through a chat interface.
    Functionality:
    - Displays the title of the application.
    - Checks for the existence of a financial database and sanitizes data if not present.
    - Configures the financial assistant if not already configured.
    - Initializes the chat history with a welcome message if not already present.
    - Handles user input through a chat interface and generates responses from the AI assistant.
    - Updates the chat history with user questions and AI responses.
    """

    st.title("AI Financial Assistant")

    # Create the financial database if it doesn't exist
    if not os.path.exists("data/financial_database.sqlite"):
        create_database()

    # Configure financial assistant if not already configured
    if "financial_assistant" not in st.session_state:
        configure_financial_assistant()

    # Initialize chat history with a welcome message
    if "chat_history" not in st.session_state:
        message = AIMessage(
            content="Welcome to the AI Financial Assistant! How can I help you?"
        )
        st.session_state.chat_history = [message]
        with st.chat_message("AI"):
            st.write(message.content)

    # User chat input
    question = st.chat_input("Ask a question")

    # Process user input and generate response if question is not empty
    if question is not None and question.strip() != "":

        # Get response from the financial assistant
        response = get_response(question, st.session_state.chat_history)

        # Update chat history with user question and AI response

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=str(response["answer"])))
        # Update chat history with DataFrame and Plotly figure if available
        st.session_state.chat_history.append(
            {"df": response["df"], "fig": response["fig"]}
        )

        show_chat_history()


if __name__ == "__main__":
    main()
