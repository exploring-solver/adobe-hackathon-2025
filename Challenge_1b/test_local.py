#!/usr/bin/env python3
"""
Local testing script for Challenge 1B
Creates sample test data and validates the solution
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys

def create_sample_pdf_content():
    """Create sample PDF content for testing"""
    # This would normally require a PDF library like reportlab
    # For testing purposes, we'll create text files that can be converted to PDFs
    
    documents = {
        "South of France - Cities.pdf": """
Comprehensive Guide to Major Cities in the South of France

Introduction
The South of France offers stunning landscapes and rich cultural heritage.

Nice: The Jewel of the French Riviera
Nice is renowned for its beautiful coastline and vibrant cultural scene.
The Promenade des Anglais is perfect for leisurely walks.
The city hosts numerous festivals throughout the year.

Marseille: A Historic Port City
Marseille is France's oldest city with a rich maritime heritage.
The Old Port has been bustling for over 2,600 years.
The city offers excellent seafood and cultural attractions.

Accommodation Options
Hotels range from luxury resorts to budget-friendly options.
Book early during peak season for better rates.
Consider location carefully for easy access to attractions.

Transportation Tips
The region is well-connected by trains and buses.
Renting a car provides flexibility for exploring.
""",
        
        "South of France - Things to Do.pdf": """
Activities and Attractions in Southern France

Coastal Adventures
The Mediterranean coastline offers numerous beach activities.
Beach hopping is popular among tourists.
Water sports include sailing, windsurfing, and diving.

Nightlife and Entertainment
The region offers vibrant nightlife with bars and clubs.
Saint-Tropez is famous for its exclusive venues.
Nice has trendy bars in the old town area.

Cultural Experiences
Visit museums and art galleries throughout the region.
Attend local festivals and cultural events.
Explore historic sites and monuments.

Group Activities
Perfect for college groups looking for adventure.
Many activities suitable for large groups of friends.
Budget-friendly options available for students.
""",
        
        "South of France - Cuisine.pdf": """
Culinary Journey Through Southern France

Local Specialties
The region is famous for bouillabaisse and ratatouille.
Fresh seafood is abundant along the coast.
Try local wines from Provence vineyards.

Restaurant Recommendations
Fine dining establishments offer Mediterranean cuisine.
Casual bistros provide authentic local experiences.
Street food markets are perfect for budget travelers.

Food Experiences
Cooking classes are available in major cities.
Wine tours provide insight into local production.
Market visits offer fresh local ingredients.

Budget Dining
Student-friendly restaurants are common.
Group dining discounts often available.
Street food provides affordable meal options.
"""
    }
    
    return documents

def create_test_environment():
    """Create a test environment with sample data"""
    # Create temporary directories
    test_dir = tempfile.mkdtemp(prefix="challenge1b_test_")
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample input configuration
    input_config = {
        "challenge_info": {
            "challenge_id": "round_1b_test",
            "test_case_name": "travel_planner_test",
            "description": "Test Case for Travel Planning"
        },
        "documents": [
            {"filename": "South of France - Cities.pdf", "title": "Cities Guide"},
            {"filename": "South of France - Things to Do.pdf", "title": "Activities Guide"},
            {"filename": "South of France - Cuisine.pdf", "title": "Food Guide"}
        ],
        "persona": {
            "role": "Travel Planner specializing in group travel for young adults"
        },
        "job_to_be_done": {
            "task": "Plan a 4-day budget-friendly trip for 10 college friends including accommodation, activities, and dining options"
        }
    }
    
    # Save input configuration
    with open(os.path.join(input_dir, "challenge1b_input.json"), 'w') as f:
        json.dump(input_config, f, indent=2)
    
    # Create sample PDF text files (in real scenario, these would be actual PDFs)
    documents = create_sample_pdf_content()
    for filename, content in documents.items():
        # For testing, save as text files (the main script can be modified to handle this)
        text_filename = filename.replace('.pdf', '.txt')
        with open(os.path.join(input_dir, text_filename), 'w') as f:
            f.write(content)
    
    return test_dir, input_dir, output_dir

def run_test():
    """Run the test scenario"""
    print("Creating test environment...")
    test_dir, input_dir, output_dir = create_test_environment()
    
    try:
        print(f"Test directory: {test_dir}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # List input files
        print("\nInput files created:")
        for file in os.listdir(input_dir):
            print(f"  - {file}")
        
        print("\nTest environment created successfully!")
        print(f"To test the solution manually:")
        print(f"1. Copy your main.py to {test_dir}")
        print(f"2. Run: python main.py --input_dir {input_dir} --output_dir {output_dir}")
        print(f"3. Check output in: {output_dir}")
        
        # Keep the test directory for manual testing
        return test_dir
        
    except Exception as e:
        print(f"Error creating test environment: {e}")
        # Clean up on error
        shutil.rmtree(test_dir, ignore_errors=True)
        return None

def validate_output(output_file):
    """Validate the output format"""
    try:
        with open(output_file, 'r') as f:
            output = json.load(f)
        
        # Check required fields
        required_fields = ['metadata', 'extracted_sections', 'subsection_analysis']
        for field in required_fields:
            if field not in output:
                print(f"Missing required field: {field}")
                return False
        
        # Check metadata
        metadata = output['metadata']
        required_metadata = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
        for field in required_metadata:
            if field not in metadata:
                print(f"Missing metadata field: {field}")
                return False
        
        # Check extracted sections
        sections = output['extracted_sections']
        if not isinstance(sections, list) or len(sections) == 0:
            print("extracted_sections should be a non-empty list")
            return False
        
        for section in sections:
            required_section_fields = ['document', 'section_title', 'importance_rank', 'page_number']
            for field in required_section_fields:
                if field not in section:
                    print(f"Missing section field: {field}")
                    return False
        
        # Check subsection analysis
        subsections = output['subsection_analysis']
        if not isinstance(subsections, list):
            print("subsection_analysis should be a list")
            return False
        
        for subsection in subsections:
            required_subsection_fields = ['document', 'refined_text', 'page_number']
            for field in required_subsection_fields:
                if field not in subsection:
                    print(f"Missing subsection field: {field}")
                    return False
        
        print("Output validation passed!")
        print(f"Found {len(sections)} extracted sections")
        print(f"Found {len(subsections)} subsections")
        return True
        
    except json.JSONDecodeError:
        print("Output file is not valid JSON")
        return False
    except FileNotFoundError:
        print("Output file not found")
        return False
    except Exception as e:
        print(f"Error validating output: {e}")
        return False

def run_docker_test(test_dir):
    """Run the solution using Docker"""
    try:
        input_dir = os.path.join(test_dir, "input")
        output_dir = os.path.join(test_dir, "output")
        
        print("Building Docker image...")
        build_cmd = [
            "docker", "build", 
            "--platform", "linux/amd64",
            "-t", "challenge1b:test",
            "."
        ]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Docker build failed: {result.stderr}")
            return False
        
        print("Running Docker container...")
        run_cmd = [
            "docker", "run", "--rm",
            "-v", f"{input_dir}:/app/input",
            "-v", f"{output_dir}:/app/output",
            "--network", "none",
            "challenge1b:test"
        ]
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Docker run failed: {result.stderr}")
            return False
        
        print("Docker execution completed successfully!")
        print(f"Output: {result.stdout}")
        
        # Validate output
        output_file = os.path.join(output_dir, "challenge1b_output.json")
        return validate_output(output_file)
        
    except subprocess.TimeoutExpired:
        print("Docker execution timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"Error running Docker test: {e}")
        return False

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--docker":
        # Run with Docker
        test_dir = run_test()
        if test_dir:
            success = run_docker_test(test_dir)
            if success:
                print("\n✅ All tests passed!")
            else:
                print("\n❌ Tests failed!")
            # Clean up
            shutil.rmtree(test_dir, ignore_errors=True)
    else:
        # Just create test environment
        test_dir = run_test()
        if test_dir:
            print(f"\n✅ Test environment created at: {test_dir}")
            print("Run with --docker flag to test with Docker")

if __name__ == "__main__":
    main()