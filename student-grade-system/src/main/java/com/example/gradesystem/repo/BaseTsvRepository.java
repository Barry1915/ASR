package com.example.gradesystem.repo;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

abstract class BaseTsvRepository<T> {
    protected final Path filePath;

    protected BaseTsvRepository(Path filePath) {
        this.filePath = filePath;
        try {
            if (Files.notExists(filePath)) {
                Files.createDirectories(filePath.getParent());
                Files.createFile(filePath);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize repository file: " + filePath, e);
        }
    }

    protected List<String> readAllLines() {
        try {
            return Files.readAllLines(filePath, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    protected void writeAllLines(List<String> lines) {
        try {
            Files.write(filePath, lines, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    protected List<T> loadAll() {
        List<String> lines = readAllLines();
        List<T> items = new ArrayList<>();
        for (String line : lines) {
            if (line.isBlank()) continue;
            items.add(parse(line));
        }
        return items;
    }

    protected void saveAll(List<T> items) {
        List<String> lines = new ArrayList<>();
        for (T t : items) {
            lines.add(serialize(t));
        }
        writeAllLines(lines);
    }

    protected abstract T parse(String line);
    protected abstract String serialize(T item);
}
