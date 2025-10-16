package com.example.gradesystem.model;

import java.util.Objects;

public class Enrollment {
    private final String studentId;
    private final String courseId;
    private final String term; // e.g., "2025-Fall", may be empty for legacy
    private Double grade; // nullable until graded

    public Enrollment(String studentId, String courseId, Double grade) {
        this(studentId, courseId, grade, "");
    }

    public Enrollment(String studentId, String courseId, Double grade, String term) {
        this.studentId = studentId;
        this.courseId = courseId;
        this.grade = grade;
        this.term = term == null ? "" : term;
    }

    public String getStudentId() { return studentId; }
    public String getCourseId() { return courseId; }
    public String getTerm() { return term; }
    public Double getGrade() { return grade; }
    public void setGrade(Double grade) { this.grade = grade; }

    @Override public String toString() { return "Enrollment{" + studentId + ", " + courseId + ", term='" + term + "', grade=" + grade + "}"; }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Enrollment)) return false;
        Enrollment that = (Enrollment) o;
        return Objects.equals(studentId, that.studentId) && Objects.equals(courseId, that.courseId) && Objects.equals(term, that.term);
    }
    @Override public int hashCode() { return Objects.hash(studentId, courseId, term); }
}
