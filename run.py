from app import application  # import the Flask app instance

if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8080, debug=True)