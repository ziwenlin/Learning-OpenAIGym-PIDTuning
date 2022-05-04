import os
import tkinter.filedialog
import tkinter.messagebox
from abc import abstractmethod
from typing import List, Callable, Dict


class Base:
    @abstractmethod
    def __init__(self, master, name):
        self.widget = master
        self.name = name


class Widget(Base):
    def __init__(self, master, name):
        super().__init__(master, name)
        self.can_restore = False
        self.can_update = False
        self.widget_var = tkinter.Variable()

    def get_value(self):
        return self.widget_var.get()

    def set_value(self, value):
        return self.widget_var.set(value)

    def toggle_variable(self):
        if type(self.widget_var) is not tkinter.Variable:
            return
        entry = self.widget
        variable = tkinter.StringVar(entry)
        entry.config(textvariable=variable)
        self.widget_var = variable


class Spacer(Base):
    def __init__(self, master, size):
        super().__init__(master, 'Spacer' + str(size))
        spacer = tkinter.Frame(master, height=size)
        spacer.pack(fill=tkinter.BOTH)
        self.widget = spacer


class Frame(Base):
    def __init__(self, master, name):
        super().__init__(master, name)
        self.widget = tkinter.Frame(master)
        self.widget.pack(fill=tkinter.BOTH)
        self.widgets: Dict[str, Widget] = {}

    def get_widget(self, search) -> Widget:
        for id, widget in self.widgets.items():
            if search in id:
                return widget

    def add_widget(self, id_name, widget):
        index = 0
        while id_name + str(index) in self.widgets:
            index += 1
        id_name += str(index)
        self.widgets.update({id_name: widget})

    def add_spacer(self, size):
        spacer = Spacer(self.widget, size)
        self.add_widget('Spacer:' + str(size), spacer)
        return spacer

    def add_label(self, text):
        label = Label(self.widget, text)
        self.add_widget('Label:' + text, label)
        return label

    def add_entry(self, name):
        entry = Entry(self.widget, name)
        self.add_widget('Entry:' + name, entry)
        return entry

    def add_button(self, name, command):
        button = Button(self.widget, name)
        button.set_command(command)
        self.add_widget('Button:' + name, button)
        return button

    def add_spinbox(self, name):
        spinbox = Spinbox(self.widget, name)
        self.add_widget('Checkbox:' + name, spinbox)
        return spinbox

    def add_checkbox(self, name):
        checkbox = Checkbox(self.widget, name)
        self.add_widget('Checkbox:' + name, checkbox)
        return checkbox


class Label(Widget):
    def __init__(self, master, text):
        super().__init__(master, text)
        label = tkinter.Label(master, text=text)
        label.config(anchor='sw', padx=5, height=1)
        label.pack(fill=tkinter.BOTH)
        self.widget = label


class Entry(Widget):
    def __init__(self, master, text):
        super().__init__(master, text)
        entry = tkinter.Entry(master)
        entry.config(width=50)
        entry.pack(fill=tkinter.BOTH)
        self.widget = entry


class Button(Widget):
    def __init__(self, master, name):
        super().__init__(master, name)
        button = tkinter.Button(master, text=name)
        button.config(height=2, anchor=tkinter.W, padx=8)
        button.pack(fill=tkinter.BOTH)
        self.widget = button

    def set_command(self, command):
        self.widget.config(command=command)


class Spinbox(Widget):
    def __init__(self, master, name):
        super().__init__(master, name)
        widget = tkinter.Spinbox(master)
        widget.pack(fill=tkinter.BOTH)
        self.widget = widget

    def increments(self, low, high, increment):
        self.widget.config(from_=low, to=high, increment=increment)


class Checkbox(Widget):
    def __init__(self, master, name):
        super().__init__(master, name)
        widget = tkinter.Checkbutton(master, text=name)
        widget.config(height=2, anchor=tkinter.W, padx=8)
        widget.pack(fill=tkinter.BOTH)
        variable = tkinter.IntVar(widget, 0)
        widget.config(variable=variable)
        self.widget_var = variable
        self.widget = widget


class FileControl(Frame):
    def __init__(self, master):
        super().__init__(master, 'FileControl')
        self.add_label('File Panel')
        self.add_button('Save', self.save)
        self.add_button('Load', self.load)
        self.add_checkbox('Load on start')
        label = self.add_label('File name')
        label.toggle_variable()
        self.add_button('Select file', self.select)

    def save(self):
        pass

    def load(self):
        pass

    def select(self):
        filename = select_file()
        if filename == '':
            return
        widget = self.get_widget('File name')
        widget.set_value(filename)


class OptionControl(Frame):
    def __init__(self, master):
        super().__init__(master, 'OptionControl')
        self.info: Dict[str, any] = {}
        self.add_label('Option Panel')
        self.add_button('Refresh', self.option_refresh)
        self.add_button('Update', self.option_update)

    def info_update(self, info):
        self.info.update(info)
        self.option_refresh()

    def option_create(self, name, value):
        if type(value) is float:
            self.add_label(name)
            widget = self.add_spinbox(name)
            widget.increments(value * -10, value * 10, value / 100)
            widget.toggle_variable()
            widget.set_value(value)
        elif type(value) is int:
            self.add_label(name)
            widget = self.add_spinbox(name)
            widget.increments(value * -10, value * 10, value // 10)
            widget.toggle_variable()
            widget.set_value(value)
        elif type(value) is str:
            self.add_label(name)
            widget = self.add_entry(name)
            widget.toggle_variable()
            widget.set_value(value)
        elif type(value) is bool:
            self.add_label(name)
            widget = self.add_checkbox(name)
            widget.set_value(value)
        elif type(value) in (dict, list):
            raise NotImplementedError(name, value)
        else:
            raise NotImplementedError(name, value)

    def option_refresh(self):
        for key_info, value in self.info.items():
            for id_widget, widget in self.widgets.items():
                if widget.name != key_info:
                    continue
                if id_widget.find('Label') >= 0:
                    continue
                widget.set_value(value)
                break
            else:
                self.option_create(key_info, value)

    def option_update(self):
        for id_widget, widget in self.widgets.items():
            if widget.name not in self.info.keys():
                continue
            if id_widget.find('Label') >= 0:
                continue
            info, id, value = self.info, widget.name, widget.get_value()
            if type(info[id]) is type(value):
                info[id] = value
            elif type(info[id]) is float:
                info[id] = float(value)
            elif type(info[id]) is int:
                info[id] = int(value)
            elif type(info[id]) is bool:
                info[id] = bool(value)
            else:
                raise NotImplementedError(id, value)


class ActionControl(Frame):
    def __init__(self, master):
        super().__init__(master, 'ActionControl')
        # self.add_spacer(10)
        self.commands: List[Callable] = [
            lambda: None for _ in range(4)]
        self.add_label('Control Panel')
        self.add_button('Start', self.start)
        self.add_button('Stop', self.stop)
        # self.add_spacer(10)
        self.add_button('Pause', self.pause)
        self.add_button('Resume', self.resume)

    def set_commands(self, commands):
        self.commands.clear()
        self.commands.extend(commands)

    def start(self):
        self.commands[0]()

    def stop(self):
        self.commands[1]()

    def pause(self):
        self.commands[2]()

    def resume(self):
        self.commands[3]()


def select_file():
    filetypes = (
        ('All files', '*.*'),
    )
    filename = tkinter.filedialog.askopenfilename(
        title='Open a file',
        initialdir=os.getcwd(),
        filetypes=filetypes)
    return filename


def main():
    import settings

    root = tkinter.Tk()
    ac = ActionControl(root)
    op = OptionControl(root)
    op.info_update(settings.get_dict())
    root.mainloop()


if __name__ == '__main__':
    main()
