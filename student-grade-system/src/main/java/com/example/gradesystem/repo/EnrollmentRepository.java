package com.example.gradesystem.repo;

import com.example.gradesystem.model.Enrollment;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class EnrollmentRepository extends BaseTsvRepository<Enrollment> {
    public EnrollmentRepository(Path filePath) { super(filePath); }

    public List<Enrollment> findAll() { return loadAll(); }

    public List<Enrollment> findByStudentId(String studentId) {
        return loadAll().stream().filter(e -> e.getStudentId().equals(studentId)).collect(Collectors.toList());
    }

    public List<Enrollment> findByStudentIdAndTerm(String studentId, String term) {
        String t = term == null ? "" : term;
        return loadAll().stream().filter(e -> e.getStudentId().equals(studentId) && e.getTerm().equals(t)).collect(Collectors.toList());
    }

    public List<Enrollment> findByCourseId(String courseId) {
        return loadAll().stream().filter(e -> e.getCourseId().equals(courseId)).collect(Collectors.toList());
    }

    public List<Enrollment> findByCourseIdAndTerm(String courseId, String term) {
        String t = term == null ? "" : term;
        return loadAll().stream().filter(e -> e.getCourseId().equals(courseId) && e.getTerm().equals(t)).collect(Collectors.toList());
    }

    public Optional<Enrollment> find(String studentId, String courseId) {
        return loadAll().stream().filter(e -> e.getStudentId().equals(studentId) && e.getCourseId().equals(courseId)).findFirst();
    }

    public Optional<Enrollment> find(String studentId, String courseId, String term) {
        String t = term == null ? "" : term;
        return loadAll().stream().filter(e -> e.getStudentId().equals(studentId) && e.getCourseId().equals(courseId) && e.getTerm().equals(t)).findFirst();
    }

    public void save(Enrollment enrollment) {
        List<Enrollment> list = new ArrayList<>(loadAll());
        list.remove(enrollment);
        list.add(enrollment);
        saveAll(list);
    }

    @Override protected Enrollment parse(String line) {
        String[] p = line.split("\t", -1);
        // Backward compatible: 3 columns -> studentId, courseId, grade; 4th optional is term
        Double grade = p.length >= 3 && p[2].isBlank() ? null : (p.length >= 3 ? tryParseDouble(p[2]) : null);
        String term = p.length >= 4 ? p[3] : "";
        return new Enrollment(p[0], p[1], grade, term);
    }

    @Override protected String serialize(Enrollment item) {
        String g = item.getGrade() == null ? "" : Double.toString(item.getGrade());
        String t = item.getTerm() == null ? "" : item.getTerm();
        return item.getStudentId() + "\t" + item.getCourseId() + "\t" + g + "\t" + t;
    }

    private static Double tryParseDouble(String s) {
        try { return Double.parseDouble(s); } catch (Exception e) { return null; }
    }
}
