package com.example.gradesystem.model;

import java.util.Objects;

public class Student {
    private final String id;
    private final String name;

    public Student(String id, String name) {
        this.id = id;
        this.name = name;
    }

    public String getId() { return id; }
    public String getName() { return name; }

    @Override public String toString() { return "Student{" + id + ", name='" + name + "'}"; }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Student)) return false;
        Student student = (Student) o;
        return Objects.equals(id, student.id);
    }
    @Override public int hashCode() { return Objects.hash(id); }
}
