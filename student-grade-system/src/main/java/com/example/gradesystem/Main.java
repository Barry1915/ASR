package com.example.gradesystem;

import com.example.gradesystem.cli.Menu;
import com.example.gradesystem.web.WebServer;

public class Main {
    public static void main(String[] args) {
        if (args.length > 0 && args[0].equalsIgnoreCase("web")) {
            new WebServer().start(8080);
        } else {
            new Menu().start();
        }
    }
}
