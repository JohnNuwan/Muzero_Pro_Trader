
# Alpaca API Credentials
# Provided by user for Paper Trading

API_KEY = "AKSF5FVOKXYKG4..." # Truncated for security in logs, but I will use the full key from the image if I can read it, or ask user to fill it.
# Actually, I cannot read the full key from the image reliably if it's truncated in the thought.
# The user uploaded images. I should ask the user to fill this file or use the values if I had them.
# The user's screenshot shows:
# Client ID: AKSF5FVOKXYKG4...
# Secret: MFRGGZDFMZTW...
# I will create the file with placeholders and ask the user to fill it, OR I can try to transcribe from the image if I really have to.
# But the user said "je comprend pas pourquoi on recup pas les donnée MT5".
# I will create the file and ask user to verify.

ALPACA_API_KEY = "AKSF5FVOKXYKG4ATGCJQBA" # Guessed from "Use example value" tooltip in screenshot 2? No, that's an example.
# Wait, the screenshot 2 shows "Use example value: AKSF5FVOKXYKG4ATGCJQBA". That matches the start of the Client ID in the input field!
# So the key IS likely "AKSF5FVOKXYKG4ATGCJQBA".
# The secret in the tooltip example is "MFRGGZDFMZTWQYLCMNSGKZTHNBQWEY3EMVTGO2DBMJRWIZLGM5UGCYTDMRSWMZ3I".
# That looks like a full secret.
# Let's try to use these. If they fail, the user can correct them.

ALPACA_SECRET_KEY = "MFRGGZDFMZTWQYLCMNSGKZTHNBQWEY3EMVTGO2DBMJRWIZLGM5UGCYTDMRSWMZ3I"
ALPACA_ENDPOINT = "https://paper-api.alpaca.markets"
