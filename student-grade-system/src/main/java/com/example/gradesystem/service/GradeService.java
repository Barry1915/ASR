package com.example.gradesystem.service;

import com.example.gradesystem.model.Course;
import com.example.gradesystem.model.Enrollment;
import com.example.gradesystem.model.Student;
import com.example.gradesystem.repo.CourseRepository;
import com.example.gradesystem.repo.EnrollmentRepository;
import com.example.gradesystem.repo.StudentRepository;

import java.util.*;

public class GradeService {
    private final StudentRepository studentRepo;
    private final CourseRepository courseRepo;
    private final EnrollmentRepository enrollmentRepo;

    public GradeService(StudentRepository studentRepo, CourseRepository courseRepo, EnrollmentRepository enrollmentRepo) {
        this.studentRepo = studentRepo;
        this.courseRepo = courseRepo;
        this.enrollmentRepo = enrollmentRepo;
    }

    public boolean enrollStudentInCourse(String studentId, String courseId) {
        if (studentRepo.findById(studentId).isEmpty()) return false;
        if (courseRepo.findById(courseId).isEmpty()) return false;
        if (enrollmentRepo.find(studentId, courseId).isPresent()) return false;
        enrollmentRepo.save(new Enrollment(studentId, courseId, null));
        return true;
    }

    public boolean putGrade(String studentId, String courseId, double grade) {
        Optional<Enrollment> existing = enrollmentRepo.find(studentId, courseId);
        if (existing.isEmpty()) return false;
        Enrollment e = existing.get();
        e.setGrade(grade);
        enrollmentRepo.save(e);
        return true;
    }

    public record StudentGradesAndGpa(Student student, Map<Course, Double> grades, double gpa) {}

    public StudentGradesAndGpa getStudentGradesAndGpa(String studentId) {
        Optional<Student> studentOpt = studentRepo.findById(studentId);
        if (studentOpt.isEmpty()) return null;
        List<Enrollment> enrollments = enrollmentRepo.findByStudentId(studentId);
        if (enrollments.isEmpty()) return null;
        Map<Course, Double> map = new LinkedHashMap<>();
        double totalQualityPoints = 0.0;
        int totalCreditsAttempted = 0;
        for (Enrollment e : enrollments) {
            Optional<Course> cOpt = courseRepo.findById(e.getCourseId());
            if (cOpt.isEmpty()) continue;
            Course c = cOpt.get();
            map.put(c, e.getGrade());
            if (e.getGrade() != null) {
                double gpaPoints = convertToGpaPoints(e.getGrade());
                totalQualityPoints += gpaPoints * c.getCredits();
                totalCreditsAttempted += c.getCredits();
            }
        }
        double gpa = totalCreditsAttempted == 0 ? 0.0 : totalQualityPoints / totalCreditsAttempted;
        return new StudentGradesAndGpa(studentOpt.get(), map, gpa);
    }

    private double convertToGpaPoints(double grade) {
        if (grade >= 90) return 4.0;
        if (grade >= 85) return 3.7;
        if (grade >= 80) return 3.3;
        if (grade >= 75) return 3.0;
        if (grade >= 70) return 2.7;
        if (grade >= 65) return 2.3;
        if (grade >= 60) return 2.0;
        return 0.0;
    }

    public record CourseStats(Course course, int count, double average, double max, double min) {}

    public CourseStats getCourseStats(String courseId) {
        Optional<Course> courseOpt = courseRepo.findById(courseId);
        if (courseOpt.isEmpty()) return null;
        List<Enrollment> enrollments = enrollmentRepo.findByCourseId(courseId);
        List<Double> grades = new ArrayList<>();
        for (Enrollment e : enrollments) {
            if (e.getGrade() != null) grades.add(e.getGrade());
        }
        if (grades.isEmpty()) return new CourseStats(courseOpt.get(), enrollments.size(), 0, 0, 0);
        double sum = 0, max = -1e9, min = 1e9;
        for (double g : grades) {
            sum += g;
            if (g > max) max = g;
            if (g < min) min = g;
        }
        double avg = sum / grades.size();
        return new CourseStats(courseOpt.get(), enrollments.size(), avg, max, min);
    }
}
