import random
MAX_LINES = 3 # its in cap so its a value that cant be changed (constint)
MAX_BET = 100
MIN_BET = 1

ROWS = 3
COLS = 3

symbol_count = {
    "A":2,
    "B":4,
    "C":6,
    "D":8
}

symbol_value = {
    "A":5,
    "B":4,
    "C":3,
    "D":2
}

def check_winning(columns,lines,bet,values):
    winnings=0
    winning_lines = []
    for line in range(lines):
        symbol = columns[0][line]
        for column in columns:
            symbol_to_check = column[line]
            if symbol == symbol_to_check:
                break
        else:
            winnings += values[symbol] * bet
            winning_lines.append(line + 1)

    return winnings, winning_lines



def get_slot_machine_spin(rows,cols,symbols):
    all_symbols = []
    for symbol, symbol_count in symbols.items():
        for mark in range(symbol_count):
            all_symbols.append(symbol)

    columns = []
    for mark1 in range(cols):
        column = []
        current_symbols = all_symbols[:] # by using (:) his way you will make a copy of the OG so if i modify the OG the copy wont be modified
        for mark2 in range(rows):
            value = random.choice(current_symbols)
            current_symbols.remove(value)
            column.append(value)

        columns.append(column)

    return columns

def print_slot_machine(columns):
    for row in range(len(columns[0])):
        for i, column in enumerate(columns):
            print(column[row], end=" | " if i != len(columns)-1 else "\n") #Simply i made an if condition in one line


def deposit():
    while True:
        amount = input("how much would you like to deposit?")
        if amount.isdigit(): #checks if the vallue is a digit or not
            amount = float(amount) # turns a value into a right number (float) for better use
            if amount > 0 : #just checking if the value is more that zero or not
                break
            else:
                print("Please enter a positive number")
        else:
            print("Please enter a number")
    return amount

def get_number_of_lines():
    while True:
        lines = input("how many lines you want to bet on? (1 - "+ str(MAX_LINES) +")")
        if lines.isdigit(): #checks if the vallue is a digit or not
            lines = int(lines) # turns a value into a right number (float) for better use
            if lines >= 1 and lines <= MAX_LINES : #just checking if the value is more that zero or not
                break
            else:
                print("Please enter a valid number\n")
        else:
            print("Please enter a number\n")
    return lines


def get_bet():
    while True:
        amount = input("how much would you like to bet? ("+str(MIN_BET)+" - "+ str(MAX_BET) +")")
        if amount.isdigit():  # checks if the vallue is a digit or not
            amount = float(amount)  # turns a value into a right number (float) for better use
            if amount >= MIN_BET and amount <= MAX_BET:  # just checking if the value is more that zero or not
                break
            else:
                print(f"Please enter a valid number between {MIN_BET} - {MAX_BET}\n")
        else:
            print("Please enter a number\n")
    return amount

def spin(balance):
    lines = get_number_of_lines()

    while True:
        bet = get_bet()
        total_bet = bet * lines

        if total_bet > balance:
            print(
                f"You don't have enough money. your current balance is ${balance} and the of your bet is ${total_bet}\n")
        else:
            break

    print(f'you are betting ${bet} on {lines} lines. total bet is ${total_bet}')

    slots = get_slot_machine_spin(ROWS, COLS, symbol_count)
    print_slot_machine(slots)
    winnings, winning_lines = check_winning(slots, lines, bet, symbol_value)
    print(f"you won {winnings} ")
    print(f"you won on lines",*winning_lines)  # this * sign will do like an iteration on all the values so its an loop replacement

    return winnings - total_bet

def main():
    balance = deposit()
    while True:
        print(f"Current balance is ${balance}")
        spinning = input("Press Enter to Spin (q to quit): ")
        if spinning == "q":
            break
        balance +=spin(balance)

    print(f"Your current balance is ${balance}")



main()