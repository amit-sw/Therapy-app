import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """Initialize Firebase if not already initialized"""
    try:
        # If Firebase is already initialized, return True
        if firebase_admin._apps:
            return True
            
        # Check if credentials exist in Streamlit secrets
        required_keys = [
            "FIREBASE_PROJECT_ID",
            "FIREBASE_PRIVATE_KEY_ID",
            "FIREBASE_PRIVATE_KEY",
            "FIREBASE_CLIENT_EMAIL",
            "FIREBASE_CLIENT_ID",
            "FIREBASE_CLIENT_X509_CERT_URL"
        ]
        
        # Verify all required keys exist and are not empty
        missing_or_empty = []
        for key in required_keys:
            if key not in st.secrets or not st.secrets[key]:
                missing_or_empty.append(key)
        
        if missing_or_empty:
            st.error("""
            Missing or empty Firebase credentials in secrets.toml. Please add the following:
            
            [firebase]
            FIREBASE_PROJECT_ID = "your-project-id"
            FIREBASE_PRIVATE_KEY_ID = "your-private-key-id"
            FIREBASE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
            FIREBASE_CLIENT_EMAIL = "your-service-account@project.iam.gserviceaccount.com"
            FIREBASE_CLIENT_ID = "your-client-id"
            FIREBASE_CLIENT_X509_CERT_URL = "https://www.googleapis.com/..."            
            """)
            st.error(f"Missing or empty credentials: {', '.join(missing_or_empty)}")
            return False
        
        try:
            # Format the private key correctly
            private_key = st.secrets["FIREBASE_PRIVATE_KEY"]
            if '\\n' not in private_key and '\n' not in private_key:
                st.error("""
                Private key is missing newline characters. It should be in this format:
                FIREBASE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\nYourKeyHere\n-----END PRIVATE KEY-----\n"
                """)
                return False
            
            if '\\n' in private_key:
                private_key = private_key.replace('\\n', '\n')
            
            # Create the credentials dictionary
            cred_dict = {
                "type": "service_account",
                "project_id": st.secrets["FIREBASE_PROJECT_ID"],
                "private_key_id": st.secrets["FIREBASE_PRIVATE_KEY_ID"],
                "private_key": private_key,
                "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
                "client_id": st.secrets["FIREBASE_CLIENT_ID"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": st.secrets["FIREBASE_CLIENT_X509_CERT_URL"]
            }
            
            # Initialize Firebase
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            
            # Verify connection by attempting to access Firestore
            db = firestore.client()
            db.collection('Test_Collection').limit(1).stream()  # Test the connection
            
            st.success("Firebase initialized and connected successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error initializing Firebase: {str(e)}")
            if "invalid_grant" in str(e).lower():
                st.error("""
                Invalid credentials. Please check that:
                1. Your service account key is correct and not expired
                2. The private key is properly formatted with newlines
                3. All credential fields match your Firebase service account
                """)
            return False
            
    except Exception as e:
        st.error(f"Unexpected error in initialize_firebase: {str(e)}")
        return False

def upload_to_firestore(data, table_name):
    """Upload data to Firestore"""
    if not firebase_admin._apps:
        return False
    
    try:
        db = firestore.client()
        # Create a collection for LLM responses
        collection_ref = db.collection(table_name)
        # Add the document with auto-generated ID
        collection_ref.add(data)
        return True
    except Exception as e:
        st.error(f"Error uploading to Firestore: {e}")
        return False

def get_all_firestore_records(table_name):
    """Retrieve all records from Firestore"""
    #st.write("Checking Firebase initialization...")
    if not firebase_admin._apps:
        st.error("Firebase not initialized")
        return None
    
    try:
        #st.write("Getting Firestore client...")
        db = firestore.client()
        collection_ref = db.collection(table_name)
        #st.write("Streaming documents...")
        docs = collection_ref.stream()
        
        records = []
        doc_count = 0
        for doc in docs:
            doc_count += 1
            data = doc.to_dict()
            data['doc_id'] = doc.id  # Include the document ID
            records.append(data)
        
        #st.write(f"Found {doc_count} documents")
        if doc_count == 0:
            st.warning(f"No records found in Firestore collection '{table_name}'")
        return records
    except Exception as e:
        st.error(f"Error retrieving data from Firestore: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None