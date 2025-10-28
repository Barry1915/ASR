package com.example.gradesystem.model;

import java.util.Objects;

public class Course {
    private final String id;
    private final String name;
    private final int credits;

    public Course(String id, String name, int credits) {
        this.id = id;
        this.name = name;
        this.credits = credits;
    }

    public String getId() { return id; }
    public String getName() { return name; }
    public int getCredits() { return credits; }

    @Override public String toString() { return "Course{" + id + ", name='" + name + "', credits=" + credits + "}"; }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Course)) return false;
        Course course = (Course) o;
        return Objects.equals(id, course.id);
    }
    @Override public int hashCode() { return Objects.hash(id); }
}
