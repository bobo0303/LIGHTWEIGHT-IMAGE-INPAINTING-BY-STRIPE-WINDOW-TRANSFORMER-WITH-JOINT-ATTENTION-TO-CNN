import termcolor
import sys
from termcolor import colored, cprint

def grey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])
def red(content):
    return termcolor.colored(content,"red",attrs=["bold"])
def green(content):
    return termcolor.colored(content,"green",attrs=["bold"])
def yellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])
def blue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])
def magenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])
def cyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])
def white(content):
    return termcolor.colored(content,"white",attrs=["bold"])

# attrs = [bold, dark, underline, blink, reverse, conealed]
# https://pypi.org/project/termcolor/

# print(grey('grey') +' '+ red('red') +' '+ green('green') +' '+ yellow('yellow') +' '+ blue('blue') +' '+ magenta('magenta') +' '+ cyan('cyan') +' '+ white('white'))

# text = colored('Hello, World!', 'red', attrs=['reverse', 'blink'])
# print(text)
# cprint('Hello, World!', 'green', 'on_red')
#
# print_red_on_cyan = lambda x: cprint(x, 'red', 'on_cyan')
# print_red_on_cyan('Hello, World!')
# print_red_on_cyan('Hello, Universe!')
#
# for i in range(10):
#     cprint(i, 'magenta', end=' ')
#
# cprint("Attention!", 'red', attrs=['bold'], file=sys.stderr)