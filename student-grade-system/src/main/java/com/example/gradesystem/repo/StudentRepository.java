package com.example.gradesystem.repo;

import com.example.gradesystem.model.Student;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class StudentRepository extends BaseTsvRepository<Student> {
    public StudentRepository(Path filePath) { super(filePath); }

    public List<Student> findAll() { return loadAll(); }

    public Optional<Student> findById(String id) {
        return loadAll().stream().filter(s -> s.getId().equals(id)).findFirst();
    }

    public void save(Student student) {
        List<Student> students = new ArrayList<>(loadAll());
        students.remove(student);
        students.add(student);
        saveAll(students);
    }

    public boolean deleteById(String id) {
        List<Student> students = new ArrayList<>(loadAll());
        boolean removed = students.removeIf(s -> s.getId().equals(id));
        if (removed) saveAll(students);
        return removed;
    }

    @Override protected Student parse(String line) {
        String[] p = line.split("\t", -1);
        return new Student(p[0], p[1]);
    }

    @Override protected String serialize(Student item) {
        return item.getId() + "\t" + item.getName();
    }
}
