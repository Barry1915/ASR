package com.example.gradesystem.model;

import java.util.Objects;

public class Enrollment {
    private final String studentId;
    private final String courseId;
    private Double grade; // nullable until graded

    public Enrollment(String studentId, String courseId, Double grade) {
        this.studentId = studentId;
        this.courseId = courseId;
        this.grade = grade;
    }

    public String getStudentId() { return studentId; }
    public String getCourseId() { return courseId; }
    public Double getGrade() { return grade; }
    public void setGrade(Double grade) { this.grade = grade; }

    @Override public String toString() { return "Enrollment{" + studentId + ", " + courseId + ", grade=" + grade + "}"; }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Enrollment)) return false;
        Enrollment that = (Enrollment) o;
        return Objects.equals(studentId, that.studentId) && Objects.equals(courseId, that.courseId);
    }
    @Override public int hashCode() { return Objects.hash(studentId, courseId); }
}
