import markdown
import pandas as pd
import datetime
import numpy as np



def get_dataset_config():
  """
  Centralized configuration for Bank Loan dataset fields.
  This single function defines both form elements and data processing.

  Returns a dictionary where each key is a field name, and each value is
  a dictionary of properties for that field.
  """
  return {
      "age": {
          "form_field": "age_field",          # HTML name attribute
          "label": "Enter Age",               # Display label
          "type": "numeric",                  # Data type
          "placeholder": "Unkown",
          "input_type": "text",               # HTML input type
          "column_name": "person_age",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
      "gender": {
          "form_field": "gender_field",
          "label": "Choose a gender",
          "type": "categorical",
          "input_type": "select",
          "column_name": "person_gender",
          "process": lambda x: x if x != "Other" else np.nan,
          "options": {
              "male": "Male",
              "female": "Female",
              "other": "Other",
          }
      },
      "education": {
          "form_field": "education_field",
          "label": "Select highest education",
          "type": "categorical",
          "input_type": "select",
          "column_name": "person_education",
          "process": lambda x: x if x != "unknown" else np.nan,
          "options": {
              "unknown": "Unknown",
              'High School': 'High School',
              'Associate': 'Associate',
              'Bachelor': 'Bachelor',
              'Master': 'Master',
              'Doctorate': 'Doctorate'}

      },
      "income": {
          "form_field": "income_field",
          "label": "Enter Annual Income ($)",               # Display label
          "placeholder": "Unkown",
          "type": "numeric",                  # Data type
          "input_type": "text",               # HTML input type
          "column_name": "person_income",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
      "employment_experience": {
          "form_field": "employment_experience_field",
          "label": "Enter Employment Experience in Years",               # Display label
          "placeholder": "Unkown",
          "type": "numeric",                  # Data type
          "input_type": "text",               # HTML input type
          "column_name": "person_emp_exp",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
      "home_ownership": {
          "form_field": "home_ownerhip",
          "label": "Select Home Ownership",
          "type": "categorical",
          "input_type": "select",
          "column_name": "person_home_ownership",
          "process": lambda x: x if x != "unknown" else np.nan,
          "options": {
              "unknown": "Unknown",
              "MORTGAGE": "Mortgage",
              "RENT": "Rent",
              "OWN": "Own",
              "OTHER": "Other",
          }
      },
      "loan_amount": {
        "form_field": "loan_amount",
        "label": "Enter Loan Amount ($)",               # Display label
        "type": "numeric",         # Data type
        "placeholder": "Unkown",
        "input_type": "text",               # HTML input type
        "column_name": "loan_amnt",               # DataFrame column name
        "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
    },
    "loan_intent": {
        "form_field": "loan_intent",
        "label": "Select Loan Intent",
        "type": "categorical",
        "input_type": "select",
        "column_name": "loan_intent",
        "process": lambda x: x if x != "unknown" else np.nan,
        "options": {
            "unknown": "Unknown",
            "PERSONAL":"Personal",
            "EDUCATION": "Education",
            "MEDICAL": "Medical",
            "VENTURE": "Venture",
            "HOME IMPROVEMENT": "Home Improvement",
            "DEBT CONSOLIDATION": "Debt Consolidation",
        }
      },
      "loan_interest_rate": {
        "form_field": "loan_interest_rate",
        "label": "Enter Loan Interest Rate (%)",               # Display label
        "type": "numeric",
        "placeholder": "Unkown",
        # Data type
        "input_type": "text",               # HTML input type
        "column_name": "loan_int_rate",               # DataFrame column name
        "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
    },
      "loan_percent_income": {
        "form_field": "loan_percent_income",
        "label": "Enter Loan's Percentage of Income (Decimanl Equivalent Form)",               # Display label
        "placeholder": "1.0 > x > 0.0",
        "type": "numeric",                  # Data type
        "input_type": "text",               # HTML input type
        "column_name": "loan_percent_income",               # DataFrame column name
        "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
    },
      "cb_person_cred_hist_length": {
        "form_field": "cb_person_cred_hist_length",
        "label": "Enter Credit Bureau's Record of Credit History Length in Years",               # Display label
        "type": "numeric",
        "placeholder": "Unkown",# Data type
        "input_type": "text",               # HTML input type
        "column_name": "cb_person_cred_hist_length",               # DataFrame column name
        "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
    },
      "credit_score": {
        "form_field": "credit_score",
        "label": "Enter Credit Score",               # Display label
        "placeholder": "Unkown",
        "type": "numeric",                  # Data type
        "input_type": "text",               # HTML input type
        "column_name": "credit_score",               # DataFrame column name
        "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
    },
      "previous_loan_defaults_on_file": {
          "form_field": "previous_loan_defaults_on_file",
          "label": "Previous Loan Defaults on File",
          "type": "categorical",
          "input_type": "select",
          "column_name": "previous_loan_defaults_on_file",
          "process": lambda x: x if x != "unknown" else np.nan,
          "options": {
              "unknown": "Unknown",
              "Yes": "Yes",
              "No": "No",
          }
  }
  }


def get_fpage_template():
  template = '''
  <!DOCTYPE html>
  <html>
  <head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bank Loan Approval Prediction</title>
  %enhanced_styling%
  </head>

  <body>
    <div class="container">
      <div class="left-column">
        <!-- Header -->
        <div class="page-header">
          <h1>Bank Loan Prediction</h1>
          <img src='https://raw.githubusercontent.com/CharlieHudson31/MLOPS/refs/heads/main/images/cover_image.png' height=200 alt='Boank Loan Prediction Cover Image'>
        </div>

        <!-- Prediction Form -->
        <div class="form-panel">
          <h2>Enter Applicant Information</h2>
          %form_section%
        </div>

        <!-- Results Section -->
        <div class="results-panel">
          %results_section%
        </div>
      </div>

      <div class="right-column" id="tableContainer">
        <!-- This space is intentionally left empty for threshold tables -->
        %threshold_tables%
      </div>
    </div>

    <script>
      /**
      * Improved toggle function with scrolling
      *
      * This function:
      * 1. Hides all tables
      * 2. If the clicked table was hidden, shows it and scrolls to it
      * 3. If the clicked table was already visible, keeps it hidden
      *
      * @param {string} tableId - The ID of the table to toggle
      */
    function toggleTable(tableId) {
      // Get the table element
      var table = document.getElementById(tableId);

      // Get all tables
      var tables = document.querySelectorAll('.table-wrapper');

      // Check if this table is currently visible (using computed style)
      var computedStyle = window.getComputedStyle(table);
      var isVisible = computedStyle.display !== 'none';

      // First, hide all tables
      tables.forEach(function(t) {
        t.style.display = 'none';
      });

      // If the table was not visible, show it and scroll to it
      if (!isVisible) {
        table.style.display = 'block';

        // Scroll the right column into view
        document.getElementById('tableContainer').scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    // If it was visible, it will remain hidden (toggled off)
  }
    </script>
  </body>
  </html>
  '''
  return template
def get_enhanced_styling():
    """
    Returns enhanced CSS styling for the Titanic prediction page.

    This function contains all the CSS styling rules for the page, organized by section.
    Each section has detailed comments explaining what the styles control and how they work.

    Returns:
    --------
    str
        A string containing all CSS styling rules wrapped in <style> tags
    """
    return '''
    <style>
      /*********************************************
       * BASE STYLES
       * These styles apply to the entire document
       *********************************************/

      /*
       * Basic body styling
       * - Sets background color to light blue-gray
       * - Sets default text color to dark gray
       * - Uses modern, sans-serif font stack
       * - Sets comfortable line spacing
       * - Removes default margins and padding
       */
      body {
        background-color: #f0f4f8;  /* Light blue-gray background */
        color: #333;                /* Dark gray text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  /* Modern font stack */
        line-height: 1.6;           /* Comfortable line spacing for readability */
        margin: 0;                  /* Remove default margin */
        padding: 0;                 /* Remove default padding */
      }

      /*********************************************
       * LAYOUT STRUCTURE
       * Controls the two-column layout of the page
       *********************************************/

      /*
       * Main container
       * - Creates a flex container for the two-column layout
       * - Ensures the container takes up at least the full viewport height
       */
      .container {
        display: flex;              /* Use flexbox for layout */
        min-height: 100vh;          /* Ensure container is at least full viewport height */
      }

      /*
       * Left column
       * - Contains the form and results
       * - Takes up 45% of the container width
       * - Has padding around all sides
       */
      .left-column {
        width: 45%;                 /* Set column width to 45% of container */
        padding: 20px;              /* Add spacing around content */
        box-sizing: border-box;     /* Include padding in width calculation */
      }

      /*
       * Right column
       * - Reserved for threshold tables
       * - Takes up 55% of the container width
       * - Has padding around all sides
       * - Position is relative for child positioning
       */
      .right-column {
        width: 55%;                 /* Set column width to 55% of container */
        padding: 20px;              /* Add spacing around content */
        box-sizing: border-box;     /* Include padding in width calculation */
        position: relative;         /* For positioning tables */
        overflow: auto;             /* Added for Scrolling */
      }
      }

      /*********************************************
       * HEADER SECTION
       * Styles for the page header and title
       *********************************************/

      /*
       * Page header container
       * - Has a blue gradient background
       * - White text
       * - Rounded corners
       * - Bottom margin to separate from content below
       * - Subtle shadow for depth
       */
      .page-header {
        background: linear-gradient(135deg, #23283a 0%, #2b5876 100%);  /* Gradient background */
        color: black;               /* black text */
        padding: 20px;              /* Internal spacing */
        border-radius: 8px;         /* Rounded corners */
        margin-bottom: 25px;        /* Space below header */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
      }

      /*
       * Main page heading (h1)
       * - No margin to avoid extra space
       * - White text color
       * - Larger font size
       * - Medium font weight
       */
      .page-header h1 {
        margin: 0;                  /* Remove default margin */
        color: black;               /* Black text */
        font-size: 32px;            /* Larger text */
        font-weight: 500;           /* Medium weight */
      }

      /*
       * Header image styling
       * - Responsive width
       * - Maintain aspect ratio
       * - Rounded corners
       * - Space above image
       */
      .page-header img {
        max-width: 100%;            /* Responsive width */
        height: auto;               /* Maintain aspect ratio */
        border-radius: 4px;         /* Rounded corners */
        margin-top: 15px;           /* Space above image */
      }

      /*********************************************
       * PANEL STYLING
       * Styles for the form and results panels
       *********************************************/

      /*
       * Form panel styling
       * - White background
       * - Rounded corners
       * - Internal padding
       * - Subtle shadow
       * - Bottom margin for spacing
       */
      .form-panel {
        background-color: white;    /* White background */
        border-radius: 8px;         /* Rounded corners */
        padding: 25px;              /* Internal spacing */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
        margin-bottom: 25px;        /* Space below panel */
      }

      /*
       * Form panel heading
       * - No top margin to avoid extra space
       * - Blue text color
       * - Bottom border for visual separation
       * - Padding below text for spacing from border
       * - Medium font weight
       */
      .form-panel h2 {
        margin-top: 0;              /* Remove top margin */
        color: #2b5876;             /* Blue heading text */
        border-bottom: 2px solid #f0f4f8;  /* Light bottom border */
        padding-bottom: 10px;       /* Space between text and border */
        font-weight: 500;           /* Medium weight */
      }

      /*
       * Results panel styling
       * - White background
       * - Rounded corners
       * - Internal padding
       * - Subtle shadow
       */
      .results-panel {
        background-color: white;    /* White background */
        border-radius: 8px;         /* Rounded corners */
        padding: 25px;              /* Internal spacing */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
      }

      /*
       * Results panel heading
       * - No top margin to avoid extra space
       * - Blue text color
       * - Medium font weight
       */
      .results-panel h2 {
        margin-top: 0;              /* Remove top margin */
        color: #2b5876;             /* Blue heading text */
        font-weight: 500;           /* Medium weight */
      }

      /*********************************************
       * FORM CONTROL STYLING
       * Styles for form inputs, dropdowns, and buttons
       *********************************************/

      /*
       * Text input styling
       * - Full width of container
       * - Internal padding
       * - Light border
       * - Rounded corners
       * - Box sizing to include padding in width
       * - Comfortable font size
       * - Top margin for spacing from label
       */
      input[type="text"] {
        width: 100%;                /* Full width */
        padding: 10px 15px;         /* Vertical and horizontal padding */
        border: 1px solid #ddd;     /* Light gray border */
        border-radius: 4px;         /* Rounded corners */
        box-sizing: border-box;     /* Include padding in width */
        font-size: 16px;            /* Comfortable font size */
        margin-top: 6px;            /* Space from label above */
      }

      /*
       * Dropdown select styling
       * - Full width of container
       * - Internal padding
       * - Light border
       * - Rounded corners
       * - White background
       * - Box sizing to include padding in width
       * - Comfortable font size
       * - Top margin for spacing from label
       * - Remove default appearance for custom styling
       * - Add custom dropdown arrow
       */
      select {
        width: 100%;                /* Full width */
        padding: 10px 15px;         /* Vertical and horizontal padding */
        border: 1px solid #ddd;     /* Light gray border */
        border-radius: 4px;         /* Rounded corners */
        background-color: white;    /* White background */
        box-sizing: border-box;     /* Include padding in width */
        font-size: 16px;            /* Comfortable font size */
        margin-top: 6px;            /* Space from label above */
        appearance: none;           /* Remove default browser styling */

        /* Add custom dropdown arrow as background image */
        background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23007CB2%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");
        background-repeat: no-repeat;
        background-position: right 10px center;  /* Position arrow on right side */
        background-size: 12px;      /* Size of arrow */
      }

      /*
       * Form group container
       * - Adds bottom margin for spacing between form elements
       */
      .form-group {
        margin-bottom: 20px;        /* Space below group */
      }

      /*
       * Form label styling
       * - Display as block for layout
       * - Bottom margin for spacing
       * - Medium font weight
       * - Blue text color for consistency
       */
      label {
        display: block;             /* Make labels block elements */
        margin-bottom: 6px;         /* Space below label */
        font-weight: 500;           /* Medium weight */
        color: #2b5876;             /* Blue text */
      }

      /*
       * Submit button styling
       * - Blue/purple gradient background
       * - White text
       * - No border
       * - Comfortable padding
       * - Larger font size
       * - Rounded corners
       * - Pointer cursor on hover
       * - Subtle shadow for depth
       * - Smooth transition for hover effects
       */
      .submit-btn {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);  /* Gradient background */
        color: white;               /* White text */
        border: none;               /* No border */
        padding: 12px 24px;         /* Vertical and horizontal padding */
        font-size: 18px;            /* Larger text */
        border-radius: 4px;         /* Rounded corners */
        cursor: pointer;            /* Hand cursor on hover */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);  /* Subtle shadow */
        transition: all 0.3s ease;  /* Smooth transition for hover effects */
      }

      /*
       * Submit button hover state
       * - Larger shadow for "lifting" effect
       * - Slight upward movement for interaction feedback
       */
      .submit-btn:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);  /* Larger shadow */
        transform: translateY(-2px);  /* Move up slightly */
      }

      /*********************************************
       * RESULTS BUTTONS STYLING
       * Styles for the model result buttons
       *********************************************/

      /*
       * Models list container
       * - Uses flexbox for layout
       * - Allows wrapping to multiple lines if needed
       * - Adds spacing between buttons
       * - Bottom margin for spacing
       */
      .models-list {
        display: flex;              /* Use flexbox for layout */
        flex-wrap: wrap;            /* Allow wrapping to multiple lines */
        gap: 10px;                  /* Spacing between buttons */
        margin-bottom: 20px;        /* Space below button group */
      }

      /*
       * Model button styling
       * - Blue/purple gradient background
       * - White text
       * - No border
       * - Comfortable padding
       * - Rounded corners
       * - Appropriate font size
       * - Pointer cursor on hover
       * - Subtle shadow for depth
       * - Smooth transition for hover effects
       * - Only take up needed space (not full width)
       * - Center text
       */
      .model-btn {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);  /* Gradient background */
        color: white;               /* White text */
        border: none;               /* No border */
        padding: 10px 16px;         /* Vertical and horizontal padding */
        border-radius: 4px;         /* Rounded corners */
        font-size: 16px;            /* Comfortable font size */
        cursor: pointer;            /* Hand cursor on hover */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
        transition: all 0.3s ease;  /* Smooth transition for hover effects */
        flex: 0 0 auto;             /* Only take up needed space */
        text-align: center;         /* Center text */
      }

      /*
       * Model button hover state
       * - Larger shadow for "lifting" effect
       * - Slight upward movement for interaction feedback
       */
      .model-btn:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Larger shadow */
        transform: translateY(-2px);  /* Move up slightly */
      }

      /*
       * Documentation button styling
       * - Blue gradient background (different from model buttons)
       * - White text
       * - No border
       * - Appropriate padding
       * - Rounded corners
       * - Comfortable font size
       * - Pointer cursor on hover
       * - Left margin for spacing
       * - Subtle shadow for depth
       * - Smooth transition for hover effects
       */
      .doc-button {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);  /* Blue gradient */
        color: white;               /* White text */
        border: none;               /* No border */
        padding: 8px 15px;          /* Vertical and horizontal padding */
        border-radius: 4px;         /* Rounded corners */
        font-size: 16px;            /* Comfortable font size */
        cursor: pointer;            /* Hand cursor on hover */
        margin-left: 10px;          /* Space to the left */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
        transition: all 0.3s ease;  /* Smooth transition for hover effects */
      }

      /*
       * Documentation button hover state
       * - Larger shadow for "lifting" effect
       * - Slight upward movement for interaction feedback
       */
      .doc-button:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Larger shadow */
        transform: translateY(-2px);  /* Move up slightly */
      }

      /*********************************************
       * MODAL STYLING
       * Styles for the documentation popup modal
       *********************************************/

      /*
       * Modal container
       * - Hidden by default
       * - Fixed position covering the entire viewport
       * - High z-index to appear above other content
       * - Semi-transparent black background overlay
       * - Scrollable if content is too tall
       */
      .modal {
        display: none;              /* Hidden by default */
        position: fixed;            /* Fixed position */
        z-index: 1000;              /* High z-index to be on top */
        left: 0;                    /* Align to left edge */
        top: 0;                     /* Align to top edge */
        width: 100%;                /* Full width */
        height: 100%;               /* Full height */
        background-color: rgba(0,0,0,0.5);  /* Semi-transparent background */
        overflow: auto;             /* Allow scrolling if needed */
      }

      /*
       * Modal content container
       * - White background
       * - Centered with margin
       * - Internal padding
       * - Rounded corners
       * - Limited width and height
       * - Scrollable if content is too tall
       * - Relative position for close button
       * - Shadow for depth
       */
      .modal-content {
        background-color: #fefefe;  /* White background */
        margin: 2% auto;            /* Center horizontally with space at top */
        padding: 20px;              /* Internal spacing */
        border-radius: 8px;         /* Rounded corners */
        width: 80%;                 /* Width relative to viewport */
        max-width: 900px;           /* Maximum width */
        max-height: 90vh;           /* Maximum height relative to viewport */
        overflow-y: auto;           /* Vertical scrolling if needed */
        position: relative;         /* For positioning close button */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);  /* Shadow for depth */
      }

      /*
       * Close button styling
       * - Absolute position in top right corner
       * - White text on blue background
       * - Bold text
       * - Pointer cursor on hover
       * - Comfortable padding
       * - Rounded corners
       * - Shadow for depth
       * - Smooth transition for hover effects
       */
      .close-button {
        position: absolute;         /* Absolute position */
        top: 15px;                  /* Top spacing */
        right: 20px;                /* Right spacing */
        color: #fff;                /* White text */
        background-color: #2b5876;  /* Blue background */
        font-size: 16px;            /* Comfortable font size */
        font-weight: bold;          /* Bold text */
        cursor: pointer;            /* Hand cursor on hover */
        padding: 5px 15px;          /* Vertical and horizontal padding */
        border-radius: 4px;         /* Rounded corners */
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* Shadow for depth */
        transition: all 0.3s ease;  /* Smooth transition for hover effects */
      }

      /*
       * Close button hover state
       * - Darker background color
       * - Larger shadow for "lifting" effect
       */
      .close-button:hover {
        background-color: #23283a;  /* Darker blue on hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);  /* Larger shadow */
      }

      /*********************************************
       * TABLE STYLING
       * Styles for the threshold tables
       *********************************************/

      /*
       * Table wrapper container
       * - Hidden by default
       * - Sticky position to stay visible when scrolling
       * - Top spacing when sticky
       */
      .table-wrapper {
        display: none;              /* Hidden by default */
        position: sticky;           /* Sticky positioning */
        top: 20px;                  /* Top spacing when sticky */
      }

      /*
       * Common styling for both table columns
       * - White background
       * - Rounded corners
       * - Subtle shadow
       * - Internal padding
       * - Bottom margin for spacing
       */
      .table1, .table2 {
        background-color: white;    /* White background */
        border-radius: 8px;         /* Rounded corners */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
        padding: 15px;              /* Internal spacing */
        margin-bottom: 20px;        /* Space below */
      }

      /*
       * Left table column
       * - Float left
       * - 45% width of container
       */
      .table1 {
        float: left;                /* Float left */
        width: 45%;                 /* 45% width */
      }

      /*
       * Right table column
       * - Float right
       * - 45% width of container
       * - Left margin for spacing from left column
       */
      .table2 {
        float: right;               /* Float right */
        width: 45%;                 /* 45% width */
        margin-left: 10px;          /* Space from left column */
      }

      /*
       * Ensemble result styling
       * - Larger font size
       * - Medium font weight
       * - Blue text color
       * - Top margin for spacing
       * - Top padding for spacing
       * - Top border for visual separation
       * - Clear both floats to appear below buttons
       */
      .ensemble-result {
        font-size: 24px;            /* Larger text */
        font-weight: 500;           /* Medium weight */
        color: #2b5876;             /* Blue text */
        margin-top: 15px;           /* Space above */
        padding: 10px;              /* Padding */
        border-top: 2px solid #f0f4f8;  /* Light top border */
        clear: both;                /* Clear floats */
      }

      /*********************************************
       * MARKDOWN CONTENT STYLING
       * Styles for the rendered markdown in the documentation modal
       *********************************************/

      /*
       * Markdown container styling
       * - Modern font stack
       * - Comfortable line height
       */
      #markdown-content {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        line-height: 1.6;           /* Comfortable line height */
      }

      /*
       * Markdown headings
       * - Blue text color
       */
      #markdown-content h1, #markdown-content h2 {
        color: #2c3e50;             /* Blue text */
      }

      /*
       * Markdown h2 headings
       * - Bottom border for visual separation
       * - Bottom padding for spacing from border
       */
      #markdown-content h2 {
        border-bottom: 1px solid #eee;  /* Light bottom border */
        padding-bottom: 5px;        /* Space between text and border */
      }

      /*
       * Markdown code blocks
       * - Light gray background
       * - Padding for spacing
       * - Rounded corners
       * - Monospace font
       */
      #markdown-content code {
        background-color: #f5f5f5;  /* Light gray background */
        padding: 2px 4px;           /* Vertical and horizontal padding */
        border-radius: 3px;         /* Rounded corners */
        font-family: monospace;     /* Monospace font */
      }

      /*
       * Markdown lists
       * - Left padding for indentation
       */
      #markdown-content ul, #markdown-content ol {
        padding-left: 25px;         /* Left padding */
      }

      /*
       * Markdown paragraphs
       * - Bottom margin for spacing
       */
      #markdown-content p {
        margin-bottom: 16px;        /* Space below */
      }

      /*
       * Markdown blockquotes
       * - Left and right padding
       * - Gray text color
       * - Left border for visual distinction
       * - Bottom margin for spacing
       */
      #markdown-content blockquote {
        padding: 0 1em;             /* Horizontal padding */
        color: #6a737d;             /* Gray text */
        border-left: 0.25em solid #dfe2e5;  /* Left border */
        margin: 0 0 16px 0;         /* Bottom margin */
      }

      /*
       * Markdown tables
       * - Collapse borders
       * - Full width
       * - Bottom margin for spacing
       */
      #markdown-content table {
        border-collapse: collapse;  /* Collapse borders */
        width: 100%;                /* Full width */
        margin-bottom: 16px;        /* Space below */
      }

      /*
       * Markdown table cells
       * - Padding for spacing
       * - Light border
       */
      #markdown-content table th, #markdown-content table td {
        padding: 6px 13px;          /* Vertical and horizontal padding */
        border: 1px solid #dfe2e5;  /* Light border */
      }

      /*
       * Markdown table alternating rows
       * - Light gray background for even rows
       */
      #markdown-content table tr:nth-child(2n) {
        background-color: #f6f8fa;  /* Light gray background */
      }

      /*********************************************
       * RESPONSIVE DESIGN
       * Adjustments for smaller screens
       *********************************************/

      /*
       * Media query for screens smaller than 1200px
       * - Changes layout from two columns to one column
       * - Makes both columns full width
       * - Changes right column positioning
       */
      @media (max-width: 1200px) {
        .container {
          flex-direction: column;   /* Stack columns vertically */
        }

        .left-column, .right-column {
          width: 100%;              /* Full width */
        }

        .right-column {
          position: static;         /* Normal positioning */
        }
      }
    </style>
    '''

def text_input(name, label, placeholder=""):
  """
  Generate HTML for a styled text input field with label.

  This function creates a form group containing a label and text input field.
  The styling is handled by CSS classes defined in get_enhanced_styling().

  Parameters:
  -----------
  name : str
  The input field name (used for form submission and label connection)
  label : str
  The descriptive text for the label
  placeholder : str, optional
  Placeholder text shown in the input when empty

  Returns:
  --------
  str
  HTML markup for the label and input field wrapped in a form group
  """
  return f'''
  <div class="form-group">
  <label for="{name}">{label}</label>
  <input type="text" id="{name}" name="{name}" placeholder="{placeholder}">
  </div>
  '''

def dropdown_select(name, label, options):
    """
    Generate HTML for a styled dropdown select with label.

    This function creates a form group containing a label and select dropdown.
    The styling is handled by CSS classes defined in get_enhanced_styling().

    Parameters:
    -----------
    name : str
        The select field name (used for form submission and label connection)
    label : str
        The descriptive text for the label
    options : dict
        Dictionary of {value: display_text} pairs for the dropdown options

    Returns:
    --------
    str
        HTML markup for the label and select dropdown wrapped in a form group
    """
    options_html = ""
    for value, text in options.items():
        options_html += f'<option value="{value}">{text}</option>\n'

    return f'''
    <div class="form-group">
      <label for="{name}">{label}</label>
      <select id="{name}" name="{name}">
        {options_html}
      </select>
    </div>
    '''

def submit_button(label="Submit"):
    """
    Generate HTML for a styled submit button.

    This function creates a submit button for a form.
    The styling is handled by CSS classes defined in get_enhanced_styling().

    Parameters:
    -----------
    label : str, optional
        The text displayed on the button (default: "Submit")

    Returns:
    --------
    str
        HTML markup for the submit button
    """
    return f'''
    <button type="submit" class="submit-btn">{label}</button>
    '''

def get_results_section():
    """
    Generate HTML for the results section with buttons.

    This function creates the results section of the page, including:
    - A heading
    - The pipeline documentation button (inserted via placeholder)
    - A row data display area (inserted via placeholder)
    - Buttons for each model's results
    - An ensemble result display

    The styling is handled by CSS classes defined in get_enhanced_styling().

    Returns:
    --------
    str
        HTML markup for the complete results section with placeholders
    """
    return '''
    <h2>Results</h2>
    %pipeline_docs%
    <h3>%row_data%</h3>

    <div class="models-list">
      <button class="model-btn" onclick="toggleTable('lgb')">
        LGB: %lgb%
      </button>

      <button class="model-btn" onclick="toggleTable('knn')">
        KNN: %knn%
      </button>

      <button class="model-btn" onclick="toggleTable('logreg')">
        LogReg: %logreg%
      </button>

      <button class="model-btn" onclick="toggleTable('ann')">
        ANN: %ann%
      </button>
    </div>

    <div class="ensemble-result">
      Ensemble: %ensemble%
    </div>
    '''

def get_threshold_tables_section():
    """
    Generate HTML for the threshold tables section.

    This function creates the table containers for all model results.
    Each model has two tables: a threshold table and a eli5 explanation table.
    By default, all tables are hidden and will only be shown when their
    corresponding button is clicked.

    The styling is handled by CSS classes defined in get_enhanced_styling().

    Returns:
    --------
    str
        HTML markup for all the model result tables with placeholders
    """
    return '''
    <!-- LGB Model Tables -->
    <div id="lgb" class='table-wrapper'>
      <div class="table1">
        <center><h2>LGB Threshold Table</h2></center>
        %lgb_table%
      </div>
      <div class="table2">
      <center><h2>LGB eli5 Explanation</h2></center>
        %lgb_explainer_table%
      </div>
    </div>

    <!-- KNN Model Tables -->
    <div id="knn" class='table-wrapper'>
      <div class="table1">
      <center><h2>KNN Threshold Table</h2></center>
        %knn_table%
      </div>
      <div class="table2">
      <center><h2>KNN Shap Explanation</h2></center>
        %knn_explainer_table%
      </div>
    </div>

    <!-- LogReg Model Tables -->
    <div id="logreg" class='table-wrapper'>
      <div class="table1">
      <center><h2>Logistic Regression Threshold Table</h2></center>
        %logreg_table%
      </div>
      <div class="table2">
      <center><h2>Logistic Regression eli5 Explanation</h2></center>
        %logreg_explainer_table%
      </div>
    </div>

    <!-- ANN Model Tables -->
    <div id="ann" class='table-wrapper'>
      <div class="table1">
      <center><h2>ANN Threshold Table</h2></center>
        %ann_table%
      </div>
      <div class="table2">
      <center><h2>ANN Shap Explanation</h2></center>
        %ann_explainer_table%
      </div>
    </div>
    '''

def get_pipeline_documentation(md_content):

    # Convert markdown to HTML using Python-Markdown
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'nl2br']
    )

    # Create complete HTML for popup with styled content and JS functionality
    documentation_html = f"""
    <!-- Button to open modal -->
    <button class="doc-button" id="openDocBtn">See Preprocessing Steps</button>

    <!-- The Modal/Popup -->
    <div id="docModal" class="modal">
        <!-- Modal content -->
        <div class="modal-content">
            <span class="close-button" id="closeModal">CLOSE</span>
            <div id="markdown-content">
                {html_content}
            </div>
        </div>
    </div>

    <!-- JavaScript for modal functionality -->
    <script>
        // Get modal elements
        const modal = document.getElementById("docModal");
        const btn = document.getElementById("openDocBtn");
        const closeBtn = document.getElementById("closeModal");

        // Open modal when button is clicked
        btn.onclick = function() {{
            modal.style.display = "block";
        }}

        // Close modal ONLY when CLOSE is clicked
        closeBtn.onclick = function() {{
            modal.style.display = "none";
        }}

        // Prevent modal from closing when clicking on the content
        document.querySelector(".modal-content").onclick = function(event) {{
            event.stopPropagation();
        }}
    </script>
    """

    return documentation_html

def complete_form(config, form_id="row_info", action="titanic_demo/data", method="POST"):
    """
    Generate form HTML using the dataset configuration.
    """
    form_elements = []

    # Create HTML for each field based on its type
    for field_id, field_config in config.items():
        if field_config["input_type"] == "text":
            form_elements.append(
                text_input(
                    field_config["form_field"],
                    field_config["label"],
                    field_config.get("placeholder", "")
                )
            )
        elif field_config["input_type"] == "select":
            form_elements.append(
                dropdown_select(
                    field_config["form_field"],
                    field_config["label"],
                    field_config["options"]
                )
            )

    # Build the complete form
    form_html = f'''
    <form id="{form_id}" action="{action}" method="{method}">
        <input type='hidden' id='hidden1' value='hidden value'/>
        {''.join(form_elements)}
        {submit_button("Evaluate")}<span>(~30 seconds)</span>
    </form>
    '''

    return form_html

def create_page(page, **fillers):
  new_page = page[:]  #copy
  for k,v in fillers.items():
    new_page = new_page.replace(f'%{str(k)}%', str(v))
  print("loading debug - returning new_page", flush=True)
  return new_page

def create_template_page(config, fpage_template, pipeline_docs_html, lgb_table, logreg_table, knn_table, ann_table):
    """
    Main function to generate the complete Titanic prediction page.

    This function:
    1. Gets all the components (styling, form, results, tables, docs)
    2. Replaces all placeholders in the template
    3. Returns the complete HTML page

    Parameters:
    -----------

    Returns:
    --------
    str
        The complete HTML for the Titanic prediction page
    """
    # from helpers import get_enhanced_styling # Local import to avoid circular dependency if helpers imports app

    # Compute all the pieces
    form_html = complete_form(config)
    results_section_html = get_results_section()
    threshold_tables_html = get_threshold_tables_section()
    enhanced_styling = get_enhanced_styling()

    # Replace all placeholders
    fpage = fpage_template.replace('%enhanced_styling%', enhanced_styling)
    fpage = fpage.replace('%form_section%', form_html)
    fpage = fpage.replace('%results_section%', results_section_html)
    fpage = fpage.replace('%threshold_tables%', threshold_tables_html)
    fpage = fpage.replace('%pipeline_docs%', pipeline_docs_html)

    # These threshold tables do not change so add them now
    fpage = fpage.replace('%lgb_table%', lgb_table)
    fpage = fpage.replace('%logreg_table%', logreg_table)
    fpage = fpage.replace('%knn_table%', knn_table)
    fpage = fpage.replace('%ann_table%', ann_table)

    return fpage