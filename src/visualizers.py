import os
import tkinter.filedialog
import tkinter.messagebox
from abc import abstractmethod
from typing import List, Callable


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

    def set_restore(self, value):
        if not self.can_restore:
            return
        self.widget_var.set(value)

    def get_restore(self):
        if not self.can_restore:
            return None
        return self.widget_var.get()

    def set_update(self, value):
        if not self.can_update:
            return
        self.widget_var.set(value)

    def get_update(self):
        if not self.can_update:
            return None
        return self.widget_var.get()

    def get_value(self):
        return self.widget_var.get()

    def set_value(self, value):
        return self.widget_var.set(value)


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
        self.widgets = {}

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

    def toggle_variable(self):
        entry = self.widget
        variable = tkinter.StringVar(entry)
        entry.config(textvariable=variable)
        self.widget_var = variable

    def toggle_update(self):
        self.toggle_variable()
        self.can_update = True

    def toggle_restore(self):
        self.toggle_variable()
        self.can_restore = True


class Entry(Widget):
    def __init__(self, master, text):
        super().__init__(master, text)
        entry = tkinter.Entry(master)
        entry.config(width=50)
        entry.pack(fill=tkinter.BOTH)
        self.widget = entry

    def toggle_variable(self):
        entry = self.widget
        variable = tkinter.StringVar(entry)
        entry.config(textvariable=variable)
        self.widget_var = variable

    def toggle_update(self):
        self.toggle_variable()
        self.can_update = True

    def toggle_restore(self):
        self.toggle_variable()
        self.can_restore = True


class Button(Widget):
    def __init__(self, master, name):
        super().__init__(master, name)
        button = tkinter.Button(master, text=name)
        button.config(height=2, anchor=tkinter.W, padx=8)
        button.pack(fill=tkinter.BOTH)
        self.widget = button

    def set_command(self, command):
        self.widget.config(command=command)


class Checkbox(Widget):
    def __init__(self, master, name):
        super().__init__(master, name)
        widget = tkinter.Checkbutton(master, text=name)
        widget.config(height=2, anchor=tkinter.W, padx=8)
        widget.pack(fill=tkinter.BOTH)
        self.widget = widget


class FileControl(Frame):
    def __init__(self, master):
        super().__init__(master, 'FileControl')
        self.add_label('File Panel')
        self.add_button('Save', self.save)
        self.add_button('Load', self.load)
        self.add_checkbox('Load on start')
        label = self.add_label('File name')
        label.toggle_update()
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
        widget.set_update(filename)


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
    root = tkinter.Tk()
    ActionControl(root)
    FileControl(root)
    root.mainloop()


if __name__ == '__main__':
    main()
