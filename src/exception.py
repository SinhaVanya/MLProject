import sys

def error_message_detail(error, error_detail:sys):
    """
    This function takes an error and its details, extracts the file name and line number where the error occurred
    and formats a detailed error message.
    """
    ## Present in custom exception handling documentation
    _, _, exc_tb = error_detail.exc_info() # get the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename# get the file name where the error occurred
    line_number = exc_tb.tb_lineno # get the line number where the error occurred
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"# format the error message
    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It takes an error and its details and formats a detailed error message using the error_message_detail function.
    """
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) # call the parent constructor with the error message
        self.error_message = error_message_detail(error_message, error_detail) # format the error message using the custom function

    def __str__(self):
        return self.error_message # return the formatted error message when the object is printed