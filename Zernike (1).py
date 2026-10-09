def Zernike_controller(commander, command, para='c', input=None, config=None):
    """
    Controls Zernike polynomials.

    Parameters:
        commander: The instance that contains the hologram with Zernike polynomials.
        command (str): Command to be executed, can be "clean", "print", "change", "add", "times", or "return".
        para (str): The variable to be modified or retrieved, can be 'c', 'n', 'm', 'enabled'.
        input (float, optional): New value to be set, only used in "change", "add", and "times" commands.
        config (int, optional): Specifies the Zernike polynomial index to operate on or retrieve from.

    Functionality:
        "clean":
            input=None, config=None
                Resets the 'para' value of all Zernike polynomials to zero or default.
            input=None, config=N (int)
                Resets the 'para' value of the Nth Zernike polynomial to zero or default.
        "print":
            input=None, config=None
                Prints the 'para' value of all Zernike polynomials.
            input=None, config=N (int)
                Prints the 'para' value of the Nth Zernike polynomial.
        "change":
            input=X (float), config=None
                Changes the 'para' value of all Zernike polynomials to X.
            input=X (float), config=N (int)
                Changes the 'para' value of the Nth Zernike polynomial to X.
        "add":
            input=X (float), config=None
                Adds X to the 'c' value of all Zernike polynomials.
            input=X (float), config=N (int)
                Adds X to the 'c' value of the Nth Zernike polynomial.
        "times":
            input=X (float), config=None
                Multiplies the 'c' value of all Zernike polynomials by X.
            input=X (float), config=N (int)
                Multiplies the 'c' value of the Nth Zernike polynomial by X.
        "return":
            input=None, config=N (int)
                Returns the 'para' value of the Nth Zernike polynomial.

    Returns:
        float or None: The value of 'para' for the specified Zernike polynomial, or None if the command is not "return".

    Exceptions:
        Raises ValueError if the input conditions are not met.
    """
    # Get the list of Zernike polynomials from the hologram instance
    zernikes = commander.hologram.zernikes

    # Validate the 'para' parameter
    valid_params = {'c', 'n', 'm', 'enabled'}
    if para not in valid_params:
        raise ValueError("Invalid para, must be one of 'c', 'n', 'm', 'enabled'")

    # Define default values for 'para' attributes
    default_values = {
        'c': 0,
        'n': 0,
        'm': 0,
        'enabled': True
    }

    # Handle the "clean" command
    if command == "clean":
        if config is None:
            # Reset 'para' value for all Zernike polynomials
            for z in zernikes:
                setattr(z, para, default_values[para])
        elif isinstance(config, int) and 0 <= config < len(zernikes):
            # Reset 'para' value for the specified Zernike polynomial
            setattr(zernikes[config], para, default_values[para])
        else:
            raise ValueError("Invalid config for 'clean' command")
    
    # Handle the "print" command
    elif command == "print":
        if config is None:
            # Print 'para' value for all Zernike polynomials
            for id, z in enumerate(zernikes):
                print(f"The {id} index of {para} = ", getattr(z, para))
        elif isinstance(config, int) and 0 <= config < len(zernikes):
            # Print 'para' value for the specified Zernike polynomial
            print(f"The {config} index of {para} = ", getattr(zernikes[config], para))
        else:
            raise ValueError("Invalid config for 'print' command")
    
    # Handle the "change" command
    elif command == "change":
        if input is None:
            raise ValueError("Input must be provided for 'change' command")
        if config is None:
            # Change 'para' value for all Zernike polynomials
            for z in zernikes:
                setattr(z, para, input)
        elif isinstance(config, int) and 0 <= config < len(zernikes):
            # Change 'para' value for the specified Zernike polynomial
            setattr(zernikes[config], para, input)
        else:
            raise ValueError("Invalid config for 'change' command")
    
    # Handle the "add" command
    elif command == "add" and para == 'c':
        if input is None:
            raise ValueError("Input must be provided for 'add' command")
        if config is None:
            # Add input to 'c' value for all Zernike polynomials
            for z in zernikes:
                setattr(z, 'c', getattr(z, 'c') + input)
        elif isinstance(config, int) and 0 <= config < len(zernikes):
            # Add input to 'c' value for the specified Zernike polynomial
            setattr(zernikes[config], 'c', getattr(zernikes[config], 'c') + input)
        else:
            raise ValueError("Invalid config for 'add' command")
    
    # Handle the "times" command
    elif command == "times" and para == 'c':
        if input is None:
            raise ValueError("Input must be provided for 'times' command")
        if config is None:
            # Multiply 'c' value for all Zernike polynomials by input
            for z in zernikes:
                setattr(z, 'c', getattr(z, 'c') * input)
        elif isinstance(config, int) and 0 <= config < len(zernikes):
            # Multiply 'c' value for the specified Zernike polynomial by input
            setattr(zernikes[config], 'c', getattr(zernikes[config], 'c') * input)
        else:
            raise ValueError("Invalid config for 'times' command")
    
    # Handle the "return" command
    elif command == "return":
        if config is None or not isinstance(config, int) or not (0 <= config < len(zernikes)):
            raise ValueError("Invalid config for 'return' command")
        # Return the 'para' value for the specified Zernike polynomial
        return getattr(zernikes[config], para)

    # If the command does not match any valid option
    else:
        raise ValueError("Invalid command")
