using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace VertoTest.Models
{
    public class Image
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int ImageId { get; set; }
        public required string FileName { get; set; }
        public required string FilePath { get; set; }
        public string? AltText { get; set; }
        public DateTime DateUploaded { get; set; }

        [ForeignKey("StaticContent")]
        public int? StaticContentID { get; set; }
        public StaticContent? staticContent { get; set; }

    }
}
