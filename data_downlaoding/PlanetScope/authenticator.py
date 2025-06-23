import planet
import getpass


def auth_init_sample(api_key, my_app_username, my_app_password):
    # The SDK's Auth context and saved state can be initialized from a username
    # and password, or an API key.
    # Applications MUST NOT save the user's password.
    if api_key:
        plauth = planet.Auth.from_key(api_key)
    else:
        plauth = planet.Auth.from_login(my_app_username, my_app_password)

    plauth.store()  # Saves API Key to ~/.planet.json for SDK use.
    print(
        f"Sample SDK Auth context initialized with API Key, which has been saved to local disk: {plauth.value}"
    )
    return plauth.value


if __name__ == "__main__":
    api_key = input("API Key: ")
    if not api_key:
        username = input("Email: ")
        password = getpass.getpass(prompt="Password: ")
    else:
        username = None
        password = None

    auth_init_sample(api_key, username, password)